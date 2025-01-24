import os
import subprocess

def main():
    """
    프로젝트의 전체 작업을 순차적으로 실행하는 메인 스크립트.
    """
    print("===== Step 1: 데이터 수집 (_1Data.py) =====")
    try:
        subprocess.run(["python", "_1Data.py"], check=True)
        print("데이터 수집이 완료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"Step 1에서 오류 발생: {e}")
        return

    print("\n===== Step 2: 모델 학습 (_2Training.py) =====")
    try:
        subprocess.run(["python", "_2Training.py"], check=True)
        print("롱/숏 모델 학습이 완료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"Step 2에서 오류 발생: {e}")
        return

    print("\n===== Step 3: 최종 모델 학습 (_3FinalTraining.py) =====")
    try:
        subprocess.run(["python", "_3FinalTraining.py"], check=True)
        print("최종 모델 학습이 완료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"Step 3에서 오류 발생: {e}")
        return

    print("\n===== Step 4: 미래 시세 예측 (_4Prediction.py) =====")
    try:
        subprocess.run(["python", "_4Prediction.py"], check=True)
        print("미래 시세 예측이 완료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"Step 4에서 오류 발생: {e}")
        return

    print("\n===== 모든 작업이 성공적으로 완료되었습니다. =====")

if __name__ == "__main__":
    main()
