import os
os.environ["NPU_COMPLETION_CYCLES"] = "0"
import glob
from itertools import islice
import time

import cv2
import numpy as np
import onnx
import tqdm

import furiosa.runtime.session
from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod

def fastreid_preproc(img, input_size=(256, 128), mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """
    FastReID 모델을 위한 전처리 함수
    """
    if len(img.shape) == 3:
        padded_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)
    else:
        padded_img = np.zeros(input_size, dtype=np.float32)
    
    # 이미지 리사이징 - aspect ratio 유지
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    
    # 중앙 정렬을 위한 패딩
    dh = (input_size[0] - int(img.shape[0] * r)) // 2
    dw = (input_size[1] - int(img.shape[1] * r)) // 2
    padded_img[
        dh:dh + int(img.shape[0] * r),
        dw:dw + int(img.shape[1] * r)
    ] = resized_img
    
    # 정규화 적용
    padded_img = (padded_img - np.array(mean)) / np.array(std)
    
    # CHW 형식으로 변환
    padded_img = padded_img.transpose(2, 0, 1)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    
    return padded_img

def create_batch(images, batch_size=16, input_size=(256, 128)):
    """
    이미지들을 배치로 만드는 함수
    """
    batch = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        # 배치 크기에 맞게 패딩
        while len(batch_images) < batch_size:
            batch_images.append(batch_images[0])  # 첫 번째 이미지로 패딩
            
        processed_batch = np.stack([
            fastreid_preproc(cv2.imread(img), input_size)
            for img in batch_images
        ])
        batch.append(processed_batch)
    return batch

def quantize_fastreid(model_path, calibration_images, input_size=(256, 128), batch_size=16):
    """
    FastReID 모델 양자화 파이프라인
    """
    # 1. 모델 로드
    model = onnx.load_model(model_path)
    
    # 2. 모델 최적화
    model = optimize_model(model)
    
    # 3. 캘리브레이션 데이터셋 준비
    print("Preparing calibration dataset...")
    calibration_batches = create_batch(
        calibration_images,
        batch_size=batch_size,
        input_size=input_size
    )
    
    # 4. 캘리브레이션 수행
    print("Performing calibration...")
    calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)
    
    for batch in tqdm.tqdm(calibration_batches, desc="Calibration"):
        calibrator.collect_data([[batch]])
    
    ranges = calibrator.compute_range()
    
    # 5. 양자화
    model_quantized = quantize(model, ranges)
    
    # 6. 컴파일러 설정
    compiler_config = {}  # 기본 설정 사용
    
    return model_quantized, compiler_config

def test_inference(model_quantized, compiler_config, test_images, input_size=(256, 128), batch_size=16, num_batches=10):
    """
    양자화된 모델의 추론 성능 테스트
    """
    total_predictions = 0
    elapsed_time = 0
    
    test_batches = create_batch(
        list(islice(test_images, batch_size * num_batches)),
        batch_size=batch_size,
        input_size=input_size
    )
    
    with furiosa.runtime.session.create(model_quantized, compiler_config=compiler_config) as session:
        for batch in test_batches:
            start = time.perf_counter_ns()
            outputs = session.run([batch])
            elapsed_time += time.perf_counter_ns() - start
            total_predictions += batch_size
    
    # 평균 레이턴시 계산
    latency = elapsed_time / total_predictions
    return latency / 1_000_000  # ms 단위로 변환

def main():
    # 설정
    model_path = "baseline_R50.onnx"
    calibration_images = glob.glob("data/*.jpg")[:200]  # 100개 이미지로 캘리브레이션
    test_images = glob.glob("data/*.jpg")[200:]
    input_size = (256, 128)
    batch_size = 16
    
    # 양자화 수행
    model_quantized, compiler_config = quantize_fastreid(
        model_path,
        calibration_images,
        input_size,
        batch_size
    )
    onnx.save_model(model_quantized, "quantized_reid.onnx")
    
    # 추론 테스트
    avg_latency = test_inference(
        model_quantized,
        compiler_config,
        test_images,
        input_size,
        batch_size
    )
    
    print(f"Average Latency: {avg_latency} ms")

if __name__ == "__main__":
    main()
