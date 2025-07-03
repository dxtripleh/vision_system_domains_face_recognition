#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera Integration Test Script.

실제 카메라를 사용하여 얼굴인식 시스템의 기능을 테스트합니다.
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# 프로젝트 모듈
from scripts.core.validation.validate_hardware_connection import HardwareValidator
from domains.face_recognition.core.services.face_detection_service import FaceDetectionService
from domains.face_recognition.core.services.face_recognition_service import FaceRecognitionService
from scripts.core.monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class CameraIntegrationTester:
    """카메라 통합 테스트"""
    
    def __init__(self, camera_id: int = 0, test_duration: int = 30):
        self.camera_id = camera_id
        self.test_duration = test_duration
        
        # 테스트 결과 저장
        self.test_results = {
            'start_time': None,
            'end_time': None,
            'hardware_status': {},
            'detection_results': [],
            'recognition_results': [],
            'performance_metrics': {},
            'errors': [],
            'summary': {}
        }
        
        # 서비스 초기화
        self.hardware_validator = HardwareValidator()
        self.detection_service = FaceDetectionService(use_mock=False)
        self.recognition_service = FaceRecognitionService(use_mock=False)
        self.performance_monitor = PerformanceMonitor()
        
        # 결과 저장 디렉토리
        self.results_dir = Path("data/test_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 테스트 실행"""
        logger.info("🚀 카메라 통합 테스트 시작...")
        
        self.test_results['start_time'] = datetime.now().isoformat()
        
        try:
            # 1. 하드웨어 검증
            logger.info("1️⃣ 하드웨어 연결 검증...")
            self._test_hardware_connection()
            
            # 2. 카메라 기본 기능 테스트
            logger.info("2️⃣ 카메라 기본 기능 테스트...")
            self._test_camera_basic_functions()
            
            # 3. 얼굴 검출 테스트
            logger.info("3️⃣ 실시간 얼굴 검출 테스트...")
            self._test_face_detection()
            
            # 4. 얼굴 인식 테스트 (인물이 등록된 경우)
            logger.info("4️⃣ 얼굴 인식 테스트...")
            self._test_face_recognition()
            
            # 5. 성능 벤치마크
            logger.info("5️⃣ 성능 벤치마크...")
            self._test_performance_benchmark()
            
            # 6. 스트레스 테스트
            logger.info("6️⃣ 스트레스 테스트...")
            self._test_stress_conditions()
            
            # 결과 분석 및 요약
            self._analyze_results()
            
            logger.info("✅ 카메라 통합 테스트 완료!")
            
        except Exception as e:
            error_msg = f"테스트 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            self.test_results['errors'].append(error_msg)
        
        finally:
            self.test_results['end_time'] = datetime.now().isoformat()
            self._save_test_results()
        
        return self.test_results
    
    def _test_hardware_connection(self):
        """하드웨어 연결 테스트"""
        try:
            # 카메라 연결 검증
            validation_result = self.hardware_validator.validate_camera_connection(self.camera_id)
            self.test_results['hardware_status'] = validation_result
            
            if not validation_result['camera_available']:
                raise RuntimeError("카메라를 사용할 수 없습니다")
            
            logger.info(f"  ✅ 카메라 {self.camera_id} 연결 확인")
            logger.info(f"  📹 해상도: {validation_result['resolution']}")
            logger.info(f"  🎥 FPS: {validation_result['fps']}")
            
        except Exception as e:
            error_msg = f"하드웨어 검증 실패: {str(e)}"
            self.test_results['errors'].append(error_msg)
            raise
    
    def _test_camera_basic_functions(self):
        """카메라 기본 기능 테스트"""
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_id)
            
            if not cap.isOpened():
                raise RuntimeError("카메라를 열 수 없습니다")
            
            # 해상도 설정 테스트
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # 프레임 캡처 테스트
            frame_count = 0
            test_frames = 10
            
            for i in range(test_frames):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_count += 1
                    # 첫 번째 프레임 저장
                    if i == 0:
                        test_image_path = self.results_dir / "test_frame.jpg"
                        cv2.imwrite(str(test_image_path), frame)
                
                time.sleep(0.1)
            
            success_rate = frame_count / test_frames
            logger.info(f"  📊 프레임 캡처 성공률: {success_rate:.1%}")
            
            if success_rate < 0.8:
                self.test_results['errors'].append(f"낮은 프레임 캡처 성공률: {success_rate:.1%}")
            
        except Exception as e:
            error_msg = f"카메라 기본 기능 테스트 실패: {str(e)}"
            self.test_results['errors'].append(error_msg)
            raise
        
        finally:
            if cap:
                cap.release()
    
    def _test_face_detection(self):
        """얼굴 검출 테스트"""
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            detection_count = 0
            total_frames = 0
            processing_times = []
            
            logger.info("  👤 얼굴을 카메라 앞에 위치시켜 주세요...")
            
            start_time = time.time()
            while time.time() - start_time < 10:  # 10초간 테스트
                ret, frame = cap.read()
                if not ret:
                    continue
                
                total_frames += 1
                
                # 얼굴 검출
                detection_start = time.perf_counter()
                result = self.detection_service.detect_faces(frame)
                processing_time = (time.perf_counter() - detection_start) * 1000
                
                processing_times.append(processing_time)
                
                if result.face_count > 0:
                    detection_count += 1
                    
                    # 첫 번째 검출 시 이미지 저장
                    if detection_count == 1:
                        # 검출된 얼굴에 박스 그리기
                        annotated_frame = frame.copy()
                        for face in result.faces:
                            x, y, w, h = face.bbox
                            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"Confidence: {face.confidence:.2f}", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        detection_image_path = self.results_dir / "face_detection_result.jpg"
                        cv2.imwrite(str(detection_image_path), annotated_frame)
                        logger.info(f"    💾 검출 결과 저장: {detection_image_path}")
                
                # 실시간 피드백
                if total_frames % 30 == 0:  # 30프레임마다
                    current_rate = detection_count / total_frames if total_frames > 0 else 0
                    logger.info(f"    📈 현재 검출률: {current_rate:.1%} ({detection_count}/{total_frames})")
            
            # 결과 저장
            detection_rate = detection_count / total_frames if total_frames > 0 else 0
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            
            detection_result = {
                'total_frames': total_frames,
                'detection_count': detection_count,
                'detection_rate': detection_rate,
                'avg_processing_time_ms': avg_processing_time,
                'max_processing_time_ms': max(processing_times) if processing_times else 0,
                'min_processing_time_ms': min(processing_times) if processing_times else 0
            }
            
            self.test_results['detection_results'] = detection_result
            
            logger.info(f"  📊 검출률: {detection_rate:.1%}")
            logger.info(f"  ⏱️ 평균 처리 시간: {avg_processing_time:.1f}ms")
            
            if detection_rate < 0.3:
                self.test_results['errors'].append(f"낮은 얼굴 검출률: {detection_rate:.1%}")
            
        except Exception as e:
            error_msg = f"얼굴 검출 테스트 실패: {str(e)}"
            self.test_results['errors'].append(error_msg)
            logger.error(error_msg)
        
        finally:
            if cap:
                cap.release()
    
    def _test_face_recognition(self):
        """얼굴 인식 테스트"""
        # Mock 모드에서는 인식 테스트 스킵
        if self.recognition_service.use_mock:
            logger.info("  ⚠️ Mock 모드에서는 얼굴 인식 테스트를 건너뜁니다")
            return
        
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            recognition_attempts = 0
            recognition_success = 0
            processing_times = []
            
            logger.info("  🧑‍💼 등록된 인물의 얼굴을 카메라 앞에 위치시켜 주세요...")
            
            start_time = time.time()
            while time.time() - start_time < 15:  # 15초간 테스트
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # 얼굴 검출
                detection_result = self.detection_service.detect_faces(frame)
                
                if detection_result.face_count > 0:
                    for face in detection_result.faces:
                        recognition_attempts += 1
                        
                        # 얼굴 인식
                        recognition_start = time.perf_counter()
                        identified_person = self.recognition_service.identify_face(face)
                        processing_time = (time.perf_counter() - recognition_start) * 1000
                        
                        processing_times.append(processing_time)
                        
                        if identified_person:
                            recognition_success += 1
                            logger.info(f"    ✅ 인식 성공: {identified_person.name}")
                
                time.sleep(0.1)
            
            # 결과 저장
            recognition_rate = recognition_success / recognition_attempts if recognition_attempts > 0 else 0
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            
            recognition_result = {
                'recognition_attempts': recognition_attempts,
                'recognition_success': recognition_success,
                'recognition_rate': recognition_rate,
                'avg_processing_time_ms': avg_processing_time
            }
            
            self.test_results['recognition_results'] = recognition_result
            
            logger.info(f"  📊 인식률: {recognition_rate:.1%}")
            logger.info(f"  ⏱️ 평균 처리 시간: {avg_processing_time:.1f}ms")
            
        except Exception as e:
            error_msg = f"얼굴 인식 테스트 실패: {str(e)}"
            self.test_results['errors'].append(error_msg)
            logger.error(error_msg)
        
        finally:
            if cap:
                cap.release()
    
    def _test_performance_benchmark(self):
        """성능 벤치마크 테스트"""
        try:
            self.performance_monitor.start_monitoring()
            
            cap = cv2.VideoCapture(self.camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_times = []
            fps_values = []
            
            logger.info("  🏃‍♂️ 성능 벤치마크 실행 중...")
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 10:  # 10초간 벤치마크
                frame_start = time.perf_counter()
                
                ret, frame = cap.read()
                if ret:
                    # 얼굴 검출 수행
                    self.detection_service.detect_faces(frame)
                    frame_count += 1
                
                frame_end = time.perf_counter()
                frame_time = frame_end - frame_start
                frame_times.append(frame_time)
                
                # FPS 계산
                if frame_count > 1:
                    fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_values.append(fps)
                    
                    # 성능 메트릭 기록
                    self.performance_monitor.record_vision_metrics(
                        fps=fps,
                        processing_time_ms=frame_time * 1000,
                        detection_count=1
                    )
            
            cap.release()
            
            # 성능 통계 계산
            avg_fps = np.mean(fps_values) if fps_values else 0
            avg_frame_time = np.mean(frame_times) if frame_times else 0
            
            performance_result = {
                'avg_fps': avg_fps,
                'max_fps': max(fps_values) if fps_values else 0,
                'min_fps': min(fps_values) if fps_values else 0,
                'avg_frame_time_ms': avg_frame_time * 1000,
                'total_frames': frame_count
            }
            
            self.test_results['performance_metrics'] = performance_result
            
            logger.info(f"  📈 평균 FPS: {avg_fps:.1f}")
            logger.info(f"  ⏱️ 평균 프레임 시간: {avg_frame_time*1000:.1f}ms")
            
            self.performance_monitor.stop_monitoring()
            
        except Exception as e:
            error_msg = f"성능 벤치마크 실패: {str(e)}"
            self.test_results['errors'].append(error_msg)
            logger.error(error_msg)
    
    def _test_stress_conditions(self):
        """스트레스 조건 테스트"""
        try:
            logger.info("  💪 스트레스 테스트 실행 중...")
            
            cap = cv2.VideoCapture(self.camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # 연속 처리 테스트 (30초간)
            stress_duration = 30
            start_time = time.time()
            frame_count = 0
            error_count = 0
            
            while time.time() - start_time < stress_duration:
                try:
                    ret, frame = cap.read()
                    if ret:
                        # 얼굴 검출 연속 수행
                        self.detection_service.detect_faces(frame)
                        frame_count += 1
                    else:
                        error_count += 1
                
                except Exception as e:
                    error_count += 1
                    logger.warning(f"    ⚠️ 프레임 처리 오류: {str(e)}")
            
            cap.release()
            
            # 스트레스 테스트 결과
            error_rate = error_count / (frame_count + error_count) if (frame_count + error_count) > 0 else 0
            
            stress_result = {
                'duration_seconds': stress_duration,
                'total_frames': frame_count,
                'error_count': error_count,
                'error_rate': error_rate,
                'avg_fps_under_stress': frame_count / stress_duration
            }
            
            self.test_results['stress_test'] = stress_result
            
            logger.info(f"  📊 스트레스 테스트 - 처리 프레임: {frame_count}")
            logger.info(f"  ❌ 오류율: {error_rate:.1%}")
            
            if error_rate > 0.05:  # 5% 이상 오류
                self.test_results['errors'].append(f"높은 스트레스 테스트 오류율: {error_rate:.1%}")
            
        except Exception as e:
            error_msg = f"스트레스 테스트 실패: {str(e)}"
            self.test_results['errors'].append(error_msg)
            logger.error(error_msg)
    
    def _analyze_results(self):
        """결과 분석 및 요약"""
        summary = {
            'overall_status': 'PASS',
            'test_duration': None,
            'hardware_ok': False,
            'detection_ok': False,
            'recognition_ok': False,
            'performance_ok': False,
            'recommendations': []
        }
        
        # 테스트 지속 시간 계산
        if self.test_results['start_time'] and self.test_results['end_time']:
            start = datetime.fromisoformat(self.test_results['start_time'])
            end = datetime.fromisoformat(self.test_results['end_time'])
            summary['test_duration'] = (end - start).total_seconds()
        
        # 하드웨어 상태 확인
        hardware_status = self.test_results.get('hardware_status', {})
        summary['hardware_ok'] = hardware_status.get('camera_available', False)
        
        # 검출 성능 확인
        detection_results = self.test_results.get('detection_results', {})
        detection_rate = detection_results.get('detection_rate', 0)
        summary['detection_ok'] = detection_rate >= 0.3
        
        # 인식 성능 확인 (Mock 모드가 아닌 경우)
        if not self.recognition_service.use_mock:
            recognition_results = self.test_results.get('recognition_results', {})
            recognition_rate = recognition_results.get('recognition_rate', 0)
            summary['recognition_ok'] = recognition_rate >= 0.5
        else:
            summary['recognition_ok'] = True  # Mock 모드에서는 Pass
        
        # 성능 확인
        performance_metrics = self.test_results.get('performance_metrics', {})
        avg_fps = performance_metrics.get('avg_fps', 0)
        summary['performance_ok'] = avg_fps >= 10
        
        # 전체 상태 결정
        if self.test_results['errors'] or not all([
            summary['hardware_ok'],
            summary['detection_ok'],
            summary['recognition_ok'],
            summary['performance_ok']
        ]):
            summary['overall_status'] = 'FAIL'
        
        # 권장사항 생성
        if not summary['hardware_ok']:
            summary['recommendations'].append("카메라 연결을 확인하세요")
        
        if not summary['detection_ok']:
            summary['recommendations'].append("조명 조건을 개선하거나 카메라 위치를 조정하세요")
        
        if not summary['performance_ok']:
            summary['recommendations'].append("시스템 리소스를 확인하고 불필요한 프로세스를 종료하세요")
        
        if avg_fps < 15:
            summary['recommendations'].append("해상도를 낮추거나 GPU 가속을 활성화하세요")
        
        self.test_results['summary'] = summary
    
    def _save_test_results(self):
        """테스트 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"camera_test_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 테스트 결과 저장: {results_file}")
    
    def print_summary(self):
        """테스트 요약 출력"""
        summary = self.test_results.get('summary', {})
        
        print("\n" + "="*60)
        print("🎯 카메라 통합 테스트 결과 요약")
        print("="*60)
        
        status_emoji = "✅" if summary.get('overall_status') == 'PASS' else "❌"
        print(f"{status_emoji} 전체 상태: {summary.get('overall_status', 'UNKNOWN')}")
        
        if summary.get('test_duration'):
            print(f"⏱️ 테스트 시간: {summary['test_duration']:.1f}초")
        
        print(f"🔧 하드웨어: {'✅ OK' if summary.get('hardware_ok') else '❌ FAIL'}")
        print(f"👤 얼굴 검출: {'✅ OK' if summary.get('detection_ok') else '❌ FAIL'}")
        print(f"🧑‍💼 얼굴 인식: {'✅ OK' if summary.get('recognition_ok') else '❌ FAIL'}")
        print(f"🚀 성능: {'✅ OK' if summary.get('performance_ok') else '❌ FAIL'}")
        
        # 상세 메트릭
        if self.test_results.get('detection_results'):
            detection = self.test_results['detection_results']
            print(f"\n📊 검출 성능:")
            print(f"   - 검출률: {detection.get('detection_rate', 0):.1%}")
            print(f"   - 평균 처리 시간: {detection.get('avg_processing_time_ms', 0):.1f}ms")
        
        if self.test_results.get('performance_metrics'):
            performance = self.test_results['performance_metrics']
            print(f"\n📈 전체 성능:")
            print(f"   - 평균 FPS: {performance.get('avg_fps', 0):.1f}")
            print(f"   - 최대 FPS: {performance.get('max_fps', 0):.1f}")
        
        # 오류 및 권장사항
        if self.test_results.get('errors'):
            print(f"\n❌ 발견된 문제점:")
            for error in self.test_results['errors']:
                print(f"   - {error}")
        
        if summary.get('recommendations'):
            print(f"\n💡 권장사항:")
            for rec in summary['recommendations']:
                print(f"   - {rec}")
        
        print("="*60)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="카메라 통합 테스트")
    parser.add_argument(
        "--camera-id", 
        type=int, 
        default=0,
        help="카메라 ID (기본값: 0)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30,
        help="테스트 지속 시간 (초, 기본값: 30)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="빠른 테스트 모드 (기본 기능만)"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)8s] %(message)s'
    )
    
    # 테스트 실행
    tester = CameraIntegrationTester(
        camera_id=args.camera_id,
        test_duration=args.duration
    )
    
    try:
        print(f"\n🎬 카메라 {args.camera_id} 통합 테스트를 시작합니다...")
        print("📋 테스트 항목:")
        print("   1. 하드웨어 연결 검증")
        print("   2. 카메라 기본 기능")
        print("   3. 얼굴 검출 테스트")
        print("   4. 얼굴 인식 테스트")
        print("   5. 성능 벤치마크")
        if not args.quick:
            print("   6. 스트레스 테스트")
        print()
        
        if args.quick:
            # 빠른 테스트 모드
            tester._test_hardware_connection()
            tester._test_camera_basic_functions()
            tester._test_face_detection()
            tester._analyze_results()
        else:
            # 전체 테스트
            tester.run_comprehensive_test()
        
        # 결과 요약 출력
        tester.print_summary()
        
        # 성공/실패 반환
        summary = tester.test_results.get('summary', {})
        return 0 if summary.get('overall_status') == 'PASS' else 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 테스트가 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 