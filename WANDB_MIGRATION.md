# Neptune → WandB 마이그레이션 완료

## 변경 사항

Neptune 로깅 시스템이 WandB (Weights & Biases)로 완전히 전환되었습니다.

### 수정된 파일들

1. **secrete_project_config.py**
   - Neptune API 토큰 및 프로젝트 → WandB API 키 및 프로젝트로 변경
   - `entity` 필드 추가 (WandB 사용자명 또는 팀명)

2. **training/train.py**
   - `import neptune` → `import wandb`
   - `--neptune` 플래그 → `--wandb` 플래그
   - `neptune.init_run()` → `wandb.init()`
   - 모든 로깅 로직이 WandB API로 변경됨

3. **training/train_two_phase.py**
   - train.py와 동일한 변경 사항 적용

4. **training/trainer/trainer.py**
   - `self.neptune_run` → `self.wandb_run`
   - 모든 Neptune 로깅 (`neptune_run[key].append()`) → WandB 로깅 (`wandb.log()`)으로 변경

5. **.gitignore**
   - `.neptune` → `wandb/`

6. **TWO_PHASE_TRAINING_README.md**
   - `--neptune` 플래그 예시 → `--wandb` 플래그로 업데이트

## 설정 방법

### 1. WandB 설치

```bash
pip install wandb
```

### 2. WandB 로그인

```bash
wandb login
```

또는 API 키를 직접 입력:

```bash
wandb login --relogin
```

API 키는 https://wandb.ai/authorize 에서 확인할 수 있습니다.

### 3. secrete_project_config.py 설정

`secrete_project_config.py` 파일을 다음과 같이 수정하세요:

```python
# WandB Configuration
api_token = "YOUR_WANDB_API_KEY"  # WandB API 키로 교체
project = "DeepfakeDetection"      # 원하는 프로젝트 이름
entity = "YOUR_WANDB_ENTITY"       # WandB 사용자명 또는 팀명으로 교체
```

## 사용 방법

### 기본 학습 (WandB 로깅 활성화)

```bash
python training/train.py \
    --detector_path training/config/detector/xception.yaml \
    --wandb
```

### 2단계 학습 (WandB 로깅 활성화)

```bash
python training/train_two_phase.py \
    --detector_path training/config/detector/xception_twophase.yaml \
    --wandb
```

### WandB 없이 학습

WandB 로깅을 사용하지 않으려면 `--wandb` 플래그를 제거하면 됩니다:

```bash
python training/train.py \
    --detector_path training/config/detector/xception.yaml
```

## 로깅되는 메트릭

WandB에 다음 메트릭들이 자동으로 로깅됩니다:

### 학습 (Training)
- `train/loss/*` - 다양한 손실 값들
- `train/metric/*` - 정확도, AUC 등의 메트릭
- `train/learning_rate` - 학습률

### 테스트 (Testing)
- `test/{dataset}/loss/*` - 데이터셋별 테스트 손실
- `test/{dataset}/{metric}` - 데이터셋별 메트릭 (AUC, ACC, EER 등)
- `test/{dataset}/acc_real` - Real 샘플 정확도
- `test/{dataset}/acc_fake` - Fake 샘플 정확도

### Dataset Cartography (2단계 학습)
- `{phase}/cartography/avg_variability` - 평균 변동성
- `{phase}/cartography/avg_confidence` - 평균 신뢰도
- `{phase}/cartography/avg_correctness` - 평균 정확성
- `{phase}/cartography/num_samples` - 샘플 수

## WandB 대시보드 확인

학습을 시작하면 콘솔에 WandB 대시보드 URL이 출력됩니다.
해당 URL을 통해 실시간으로 학습 진행 상황을 모니터링할 수 있습니다.

예시:
```
wandb: 🚀 View run at https://wandb.ai/your-entity/DeepfakeDetection/runs/xxxxx
```

## 주의사항

1. **API 키 보안**: `secrete_project_config.py` 파일은 `.gitignore`에 포함되어 있으므로 Git에 커밋되지 않습니다. API 키를 공개 저장소에 업로드하지 않도록 주의하세요.

2. **첫 실행**: 처음 WandB를 사용한다면 `wandb login` 명령으로 로그인해야 합니다.

3. **오프라인 모드**: 인터넷 연결 없이 사용하려면:
   ```bash
   export WANDB_MODE=offline
   ```

4. **기존 Neptune 데이터**: 기존에 Neptune에 저장된 데이터는 WandB로 자동 이전되지 않습니다. 필요시 수동으로 데이터를 export/import 해야 합니다.

## 문제 해결

### WandB 초기화 실패
- API 키가 올바르게 설정되었는지 확인
- 인터넷 연결 확인
- `wandb login` 재실행

### 로그가 기록되지 않음
- `--wandb` 플래그가 포함되었는지 확인
- `secrete_project_config.py` 설정 확인

## 참고 자료

- [WandB 공식 문서](https://docs.wandb.ai/)
- [WandB Python SDK 가이드](https://docs.wandb.ai/guides/track)
- [WandB 예제](https://github.com/wandb/examples)
