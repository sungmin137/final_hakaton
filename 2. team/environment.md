# 개발 환경 및 플러그인

## 설치된 Claude Code 플러그인 (2026-09-09)
| 플러그인 | 용도 | 비고 |
|---|---|---|
| commit-commands | `/commit`, `/push`, PR 생성 | 바로 사용 가능 |
| github | GitHub MCP (이슈/PR/저장소 관리) | `GITHUB_PERSONAL_ACCESS_TOKEN` 환경변수 필요 (아래 참고) |
| pyright-lsp | Python 타입 검사·코드 인텔리전스 | pyright 1.1.411 (`uv tool install pyright`), Node 26 설치됨 |
| context7 | 라이브러리 최신 문서 조회 | 익명 사용 가능, `CONTEXT7_API_KEY` 선택 |
| claude-md-management | CLAUDE.md 감사/갱신 | 세션 교훈 누적용 |

## github 플러그인 토큰 설정 (사용자 직접)
1. GitHub → Settings → Developer settings → Personal access tokens에서 `repo` 권한 토큰 발급
2. `~/.zshrc`에 추가 후 터미널 재시작:
   ```bash
   export GITHUB_PERSONAL_ACCESS_TOKEN="발급받은_토큰"
   ```
- 토큰 없이도 `gh` CLI(sungmin137 로그인됨)로 push/PR은 가능. MCP 기능만 토큰이 필요.

## 로컬 도구
- Python 3.14 (시스템) + pandas 3.0 / scikit-learn 1.9 / LightGBM 4.6 / XGBoost 3.3 / CatBoost 1.2 / Optuna 4.9
- uv, git, gh CLI, Node 26 (pyright용)
- VSCode 1.136

## GitHub 저장소
- URL: https://github.com/sungmin137/final_hakaton (private, 기본 브랜치 `main`)
- 팀원 초대 (사용자 직접 실행, USERNAME 교체):
  ```bash
  gh api -X PUT repos/sungmin137/final_hakaton/collaborators/USERNAME -f permission=push
  ```
  또는 GitHub 웹 → Settings → Collaborators → Add people

## 플러그인 MCP 연결 상태 (2026-09-09)
- `github` MCP: `GITHUB_PERSONAL_ACCESS_TOKEN` 미설정으로 연결 실패. 토큰 설정 후 세션 재시작하면 해결.
- `context7` MCP: 키가 비어 있으면 빈 Authorization 헤더를 보내 401 발생. https://context7.com 에서 무료 API 키를 받아
  `export CONTEXT7_API_KEY="..."`를 `~/.zshrc`에 추가하면 해결. 둘 다 없어도 코드 작업에는 지장 없음.

## 백그라운드 실험 실행 규칙 (2026-09-15, 조용한 실패 재발 방지)
- 정직 CV는 반드시 `bash "4. src/common/run_cv.sh" <features> [params]`로 실행한다. 인자를 선택지와 대조하고, 시작·종료·결과를 `6. experiments/cv_runs.log`에 남기며, 실패하면 exit≠0.
- zsh에서 `for job in "a b"; do set -- $job` 같은 공백 분리 의존 코드를 쓰지 않는다(zsh는 변수를 분리하지 않음). 인자는 따로 넘긴다.
- 백그라운드로 던진 뒤에는 "시작됨" 메시지로 끝내지 말고, 출력 파일 첫 줄과 `ps`로 실제 진행을 확인한 뒤 다음 일을 한다.
- 로그 줄 수를 기다리는 대기 루프 대신, 작업 자체를 백그라운드로 실행해 종료 코드로 알림을 받는다.
