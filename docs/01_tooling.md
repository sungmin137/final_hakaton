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
