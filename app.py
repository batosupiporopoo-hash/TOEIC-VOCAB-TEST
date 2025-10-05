# app.py
# -*- coding: utf-8 -*-
import os
import io
import time
import random
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# 1秒ごとの自動リフレッシュ（カウントダウン用）
# ※ requirements.txt に streamlit-autorefresh を追加してください
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    def st_autorefresh(*args, **kwargs):
        return None

############################
# ページ設定 & スタイル
############################
st.set_page_config(page_title="TOEIC Vocab Quiz", page_icon="📝", layout="centered")

CUSTOM_CSS = """
<style>
main.block-container { max-width: 940px; }

/* 大見出し */
.mode-badge {
  display:inline-block; padding:4px 10px; border-radius: 999px;
  font-size: 12px; border:1px solid #d0d7de; background:#f6f8fa; color:#333;
}

/* 英単語を大きく太く */
.big-word {
  font-size: 48px;
  font-weight: 800;
  letter-spacing: 0.5px;
  margin: 0 0 0.25rem 0;
}
.pos-tag {
  display: inline-block;
  font-size: 14px;
  color: #555;
  background: #eef3ff;
  border: 1px solid #ccd9ff;
  padding: 2px 8px;
  border-radius: 12px;
  margin-bottom: 16px;
}

/* 選択肢 */
.choice {
  border: 1.5px solid var(--secondary-background-color);
  background: var(--background-color);
  border-radius: 12px;
  padding: 14px 16px;
  margin: 8px 0;
  cursor: pointer;
  font-size: 20px;
  line-height: 1.3;
}
.choice:hover {
  border-color: #7aa2ff;
  background: rgba(122,162,255,0.08);
}
/* 正解/誤答のハイライト */
.correct {
  background: #e6ffed !important;
  border-color: #46c06f !important;
  color: #0a5a24 !important;
}
.incorrect {
  background: #ffeaea !important;
  border-color: #ff6b6b !important;
  color: #8a1f1f !important;
}

/* 完了/時間切れバナー */
.done-banner {
  font-size: 42px;
  font-weight: 900;
  text-align: center;
  padding: 24px 12px;
  border-radius: 16px;
  border: 2px dashed #8ad;
  background: #f5fbff;
}
.timeout-banner {
  font-size: 42px;
  font-weight: 900;
  text-align: center;
  padding: 24px 12px;
  border-radius: 16px;
  border: 2px dashed #e57373;
  background: #fff5f5;
}

/* タイマー表示 */
.timer-box {
  display:flex; align-items:center; justify-content:space-between;
  border:1px solid #d0d7de; border-radius:12px; padding:8px 12px; margin:8px 0 6px 0;
  background:#f6f8fa;
  font-weight:700;
}
.timer-warn { background:#fff7e6; border-color:#ffcc80; }
.timer-urgent { background:#fff0f0; border-color:#ff8a80; animation: shake 0.7s infinite; }
@keyframes shake {
  0% { transform: translate(1px, 0); }
  20% { transform: translate(-1px, 0); }
  40% { transform: translate(1px, 0); }
  60% { transform: translate(-1px, 0); }
  80% { transform: translate(1px, 0); }
  100% { transform: translate(-1px, 0); }
}
.subtle { color:#666; font-size: 14px; }
kbd {
  padding: 2px 6px;
  border: 1px solid #8882;
  border-bottom-width: 2px;
  border-radius: 6px;
  background: #f8f8f8;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

############################
# パス周り（app.pyの隣を基準に探す）
############################
APP_DIR = Path(__file__).parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

CANDIDATES = [APP_DIR / "vocab.csv", APP_DIR / "vocab.sv", APP_DIR / "data" / "vocab.csv", APP_DIR / "data" / "vocab.sv"]
DATA_FILE_PATH = next((p for p in CANDIDATES if p.exists()), None)
if DATA_FILE_PATH is None:
    st.error("データファイルが見つかりません。`app.py` と同じフォルダ（または `data/`）に `vocab.csv`（または `vocab.sv`）を置いてください。")
    st.stop()

############################
# データ読み込み
############################
def _load_vocab_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_err = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            break
        except Exception as e:
            last_err = e
    if df is None:
        st.exception(last_err)
        st.error("vocabファイルの読み込みに失敗しました。文字コードや区切り文字をご確認ください。")
        st.stop()

    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    col_map_candidates = {
        "word": ["word", "単語", "英単語"],
        "pos": ["pos", "品詞"],
        "meaning": ["meaning", "意味", "和訳", "訳"],
        "level": ["level", "レベル"]
    }
    for target, candidates in col_map_candidates.items():
        for c in df.columns:
            if c in candidates:
                rename_map[c] = target
                break
    df = df.rename(columns=rename_map)

    # 必須列チェック（level は任意だが推奨）
    required = ["word", "pos", "meaning"]
    for col in required:
        if col not in df.columns:
            st.error(f"vocabファイルに必要な列 '{col}' が見つかりません。列順は 'word,pos,meaning[,level]' を推奨します。")
            st.stop()

    for col in ["word", "pos", "meaning"]:
        df[col] = df[col].astype(str).str.strip()
    if "level" not in df.columns:
        df["level"] = np.nan  # 未設定でも動くように

    # level は文字列化（"600","700","800","900" を推奨）
    df["level"] = df["level"].astype(str).str.extract(r"(\d{3,4})")[0]  # 数字だけ抽出
    # NaN になったものは "NA" にする（未指定）
    df["level"] = df["level"].fillna("NA")

    df = df.dropna(subset=["word", "meaning"]).reset_index(drop=True)
    return df

df = _load_vocab_safely(DATA_FILE_PATH)
total_words = len(df)

############################
# ユーティリティ
############################
def rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def wrong_indices_from_current_logs(logs: list[dict]) -> set[int]:
    return {int(x["idx"]) for x in logs if not x.get("correct", True)}

def wrong_indices_from_log_file(df: pd.DataFrame, path: Path, user_id: str | None) -> set[int]:
    """quiz_log.csv または logs/<user>_quiz_log.csv から誤答語を抽出し、(word,pos,meaning)で現在のDFにマッピング"""
    if not path.exists():
        return set()
    try:
        logs_df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        logs_df = pd.read_csv(path, encoding="utf-8")
    need_cols = {"word", "pos", "meaning", "correct"}
    if not need_cols.issubset(set(logs_df.columns)):
        return set()
    if user_id and "user_id" in logs_df.columns:
        logs_df = logs_df[logs_df["user_id"].astype(str) == str(user_id)]
    logs_wrong = logs_df[logs_df["correct"] == False][["word", "pos", "meaning"]].drop_duplicates()
    df2 = df.reset_index().rename(columns={"index": "idx"})
    merged = pd.merge(
        logs_wrong,
        df2[["idx", "word", "pos", "meaning"]],
        on=["word", "pos", "meaning"],
        how="inner"
    )
    return set(merged["idx"].tolist())

def export_words(df, indices: list[int]):
    if not indices:
        return None
    out = df.loc[indices, ["word", "pos", "meaning", "level"]].copy()
    buffer = io.StringIO()
    out.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()

def build_choices_same_pos(df_all: pd.DataFrame, correct_idx: int, n_choices: int, direction: str):
    """
    誤答肢はレベルを問わず **全体から** 正解と同じ POS のみで収集。
    足りない場合は POS 無視で補完。最後にシャッフル。
    """
    correct_pos = str(df_all.loc[correct_idx, "pos"]).strip()
    # 同POSのプール（正解以外）
    same_pos_pool = [i for i in df_all.index if i != correct_idx and str(df_all.loc[i, "pos"]).strip() == correct_pos]
    # 全体のプール（正解以外）
    any_pool = [i for i in df_all.index if i != correct_idx]

    random.shuffle(same_pos_pool)
    random.shuffle(any_pool)

    need = max(0, n_choices - 1)
    distractors = same_pos_pool[:need]
    if len(distractors) < need:
        # 同POSで不足 → 残りを全体から補完（重複除去）
        more = [i for i in any_pool if i not in distractors]
        distractors.extend(more[:(need - len(distractors))])

    def lab(i):
        return df_all.loc[i, "meaning"] if direction == "w2m" else df_all.loc[i, "word"]

    correct_label = df_all.loc[correct_idx, "meaning"] if direction == "w2m" else df_all.loc[correct_idx, "word"]
    options = [(correct_label, True)] + [(lab(i), False) for i in distractors]
    random.shuffle(options)
    return [{"label": lab, "is_correct": ok} for lab, ok in options]

def advance_question(state):
    if state.time_up:
        # タイムアップ時は進めない
        return
    if not state.queue:
        state.finished = True
        return
    state.current_idx = state.queue.pop(0)
    state.question_id += 1

    # 出題方向
    state.current_direction = random.choice(["w2m", "m2w"]) if state.direction == "mix" else state.direction

    # 選択肢数
    n_choices = max(2, min(state.n_choices, len(df)))  # 全体から引くため df の行数で上限

    # 選択肢（POS一致・全体から）
    state.choices = build_choices_same_pos(
        df_all=df,
        correct_idx=state.current_idx,
        n_choices=n_choices,
        direction=state.current_direction
    )

    # キーボード入力の持ち越し防止
    state.kbd_key = f"kbd_{state.question_id}"
    st.session_state[state.kbd_key] = ""

    # フィードバックのリセット
    state.pending_feedback = False
    state.last_selected = None
    state.correct_choice_pos = next(i for i, c in enumerate(state.choices) if c["is_correct"])

def process_answer(state, selected_pos: int):
    if state.time_up or state.finished:
        return
    is_correct = state.choices[selected_pos]["is_correct"]
    idx = state.current_idx

    state.stats_total += 1
    state.attempts[idx] = state.attempts.get(idx, 0) + 1

    # ログ
    state.logs.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "user_id": state.user_id or "",
        "qid": state.question_id,
        "idx": int(idx),
        "word": df.loc[idx, "word"],
        "pos": df.loc[idx, "pos"],
        "meaning": df.loc[idx, "meaning"],
        "level": df.loc[idx, "level"],
        "direction": state.current_direction,
        "correct": bool(is_correct),
        "attempt": int(state.attempts[idx]),
    })

    if is_correct:
        state.stats_correct += 1
        state.mastered.add(idx)
        advance_question(state)
        rerun()
    else:
        state.stats_wrong += 1
        state.pending_feedback = True
        state.last_selected = selected_pos
        state.queue.append(idx)

def start_quiz_with_pool(state, pool_indices: list[int]):
    if not pool_indices:
        st.warning("出題対象がありません。範囲や復習対象の設定を確認してください。")
        return
    indices = list(pool_indices)
    random.shuffle(indices)

    # キューと統計の初期化
    state.candidate_indices = indices.copy()
    state.queue = indices.copy()
    state.current_idx = None
    state.question_id = 0
    state.current_direction = "w2m"
    state.choices = []
    state.correct_choice_pos = None
    state.pending_feedback = False
    state.last_selected = None
    state.mastered = set()
    state.attempts = {}
    state.finished = False
    state.stats_total = 0
    state.stats_correct = 0
    state.stats_wrong = 0
    state.logs = []

    # タイマー初期化（語数×2秒）
    total_q = len(indices)
    state.time_limit_sec = int(total_q * 4)
    state.end_time = time.time() + state.time_limit_sec
    state.time_up = False
    state.quiz_running = True

    advance_question(state)
    rerun()

############################
# 状態初期化
############################
if "init" not in st.session_state:
    st.session_state.init = True

    # モード
    st.session_state.mode = "normal"  # 'normal' | 'review'

    # レベル & 範囲
    st.session_state.level_filter = "指定なし"  # '指定なし' | '600' | '700' | '800' | '900'
    st.session_state.range_start_all = 1
    st.session_state.range_end_all = len(df)
    # レベルごとの範囲（1ベース）
    for lv in ["600","700","800","900"]:
        cnt = len(df[df["level"] == lv])
        st.session_state[f"range_start_{lv}"] = 1
        st.session_state[f"range_end_{lv}"] = max(1, cnt)

    # 出題設定
    st.session_state.n_choices = 4
    st.session_state.direction = "w2m"  # 'w2m' | 'm2w' | 'mix'

    # 復習モード設定
    st.session_state.review_include_current_wrong = True
    st.session_state.review_include_starred = True
    st.session_state.review_include_past_log = False

    # ユーザー識別（任意）
    st.session_state.user_id = ""

    # クイズ状態
    st.session_state.queue = []
    st.session_state.candidate_indices = []
    st.session_state.current_idx = None
    st.session_state.question_id = 0
    st.session_state.current_direction = "w2m"
    st.session_state.choices = []
    st.session_state.correct_choice_pos = None
    st.session_state.kbd_key = "kbd_0"
    st.session_state.pending_feedback = False
    st.session_state.last_selected = None
    st.session_state.mastered = set()
    st.session_state.attempts = {}
    st.session_state.finished = False
    st.session_state.starred = set()
    st.session_state.logs = []
    st.session_state.stats_total = 0
    st.session_state.stats_correct = 0
    st.session_state.stats_wrong = 0

    # タイマー
    st.session_state.time_limit_sec = 0
    st.session_state.end_time = 0.0
    st.session_state.time_up = False
    st.session_state.quiz_running = False

state = st.session_state

############################
# サイドバー（設定）
############################
with st.sidebar:
    st.header("⚙️ 設定")
    st.caption(f"データ: **{DATA_FILE_PATH.name}**（{total_words}語）")

    # モード
    state.mode = st.radio(
        "モード",
        options=["normal", "review"],
        format_func=lambda x: "通常モード（範囲から出題）" if x=="normal" else "復習モード（誤答/スター）",
        index=0 if state.mode=="normal" else 1
    )

    # ユーザーID（任意）
    state.user_id = st.text_input("ユーザーID（任意）", value=state.user_id, placeholder="例）ryota.yamada")

    if state.mode == "normal":
        # レベル選択
        st.write("**レベル選択（任意）**")
        level_options = ["指定なし", "600", "700", "800", "900"]
        state.level_filter = st.selectbox("レベル", options=level_options, index=level_options.index(state.level_filter))

        if state.level_filter == "指定なし":
            # 全体から範囲
            c1, c2 = st.columns(2)
            with c1:
                start = st.number_input("開始（1〜）", min_value=1, max_value=total_words, value=state.range_start_all, step=1, key="inp_start_all")
            with c2:
                end = st.number_input("終了", min_value=1, max_value=total_words, value=state.range_end_all, step=1, key="inp_end_all")
            if start > end:
                st.warning("開始は終了以下にしてください。")
            else:
                state.range_start_all, state.range_end_all = int(start), int(end)
            st.caption("レベル未指定：全体から出題します。")
        else:
            lv = state.level_filter
            df_lv = df[df["level"] == lv]
            cnt_lv = len(df_lv)
            st.caption(f"レベル {lv}: {cnt_lv} 語")
            c1, c2 = st.columns(2)
            with c1:
                start_lv = st.number_input(f"開始（レベル{lv}内・1〜）", min_value=1, max_value=max(1, cnt_lv), value=state.__getattr__(f"range_start_{lv}"), step=1, key=f"inp_start_{lv}")
            with c2:
                end_lv = st.number_input("終了", min_value=1, max_value=max(1, cnt_lv), value=state.__getattr__(f"range_end_{lv}"), step=1, key=f"inp_end_{lv}")
            if start_lv > end_lv:
                st.warning("開始は終了以下にしてください。")
            else:
                state.__setattr__(f"range_start_{lv}", int(start_lv))
                state.__setattr__(f"range_end_{lv}", int(end_lv))
            st.caption("選択したレベル内のみから出題します。")

    else:
        st.write("**復習対象の選択**")
        state.review_include_current_wrong = st.checkbox("直近セッションの誤答を含める", value=state.review_include_current_wrong)
        state.review_include_starred = st.checkbox("⭐スター語を含める", value=state.review_include_starred)
        state.review_include_past_log = st.checkbox("過去ログの誤答も含める（ユーザーIDがあればそのユーザーのログ）", value=state.review_include_past_log)
        # プレビュー
        preview_sets = []
        if state.review_include_current_wrong:
            preview_sets.append(("直近誤答", len(wrong_indices_from_current_logs(state.logs))))
        if state.review_include_starred:
            preview_sets.append(("スター", len(state.starred)))
        if state.review_include_past_log:
            # ユーザーIDがあれば個人ログを使う
            path = LOG_DIR / f"{state.user_id}_quiz_log.csv" if state.user_id else APP_DIR / "quiz_log.csv"
            preview_sets.append(("過去ログ誤答", len(wrong_indices_from_log_file(df, path, state.user_id))))
        if preview_sets:
            st.caption("候補数プレビュー: " + " / ".join([f"{k}:{v}" for k, v in preview_sets]))

    # 共通の出題設定
    state.n_choices = st.slider("選択肢の数", min_value=2, max_value=8, value=state.n_choices, step=1)
    state.direction = st.selectbox(
        "出題方向",
        options=[("w2m", "単語 → 和訳"), ("m2w", "和訳 → 単語"), ("mix", "ミックス")],
        index={"w2m": 0, "m2w": 1, "mix": 2}[state.direction],
        format_func=lambda x: x[1]
    )[0]

    st.write("---")
    st.caption("Tips: キーボードで素早く選択できます（1〜9 / A〜J）。")

    if st.button("▶️ クイズを開始 / リセット", use_container_width=True):
        if state.mode == "normal":
            if state.level_filter == "指定なし":
                s0 = max(1, min(state.range_start_all, total_words)) - 1
                e0 = max(1, min(state.range_end_all, total_words)) - 1
                pool = list(range(s0, e0 + 1))
            else:
                lv = state.level_filter
                df_lv = df[df["level"] == lv].reset_index()  # index列に元のidx
                cnt = len(df_lv)
                s_lv = max(1, min(state.__getattr__(f"range_start_{lv}"), cnt)) - 1
                e_lv = max(1, min(state.__getattr__(f"range_end_{lv}"), cnt)) - 1
                selected = df_lv.loc[s_lv:e_lv, "index"].tolist()  # 元の idx に戻す
                pool = selected
            start_quiz_with_pool(state, pool)
        else:
            pool_set = set()
            if state.review_include_current_wrong:
                pool_set |= wrong_indices_from_current_logs(state.logs)
            if state.review_include_starred:
                pool_set |= set(state.starred)
            if state.review_include_past_log:
                # ユーザーIDがあれば個別ログを優先
                log_path = LOG_DIR / f"{state.user_id}_quiz_log.csv" if state.user_id else APP_DIR / "quiz_log.csv"
                pool_set |= wrong_indices_from_log_file(df, log_path, state.user_id)
            start_quiz_with_pool(state, sorted(pool_set))

############################
# ヘッダー
############################
st.title("📝 TOEIC Vocabulary Quiz")
badge = "通常モード" if state.mode == "normal" else "復習モード（誤答/スター）"
st.markdown(f'<span class="mode-badge">{badge}</span>', unsafe_allow_html=True)
st.caption("正解で即次へ。誤答は正解を1秒ハイライトした後に後回し。全問1回は正答するまで続きます。タイムアップにも注意！")

############################
# 未開始時
############################
if not state.queue and state.current_idx is None and not state.finished and not state.time_up:
    if state.mode == "normal":
        st.info("左の **「▶️ クイズを開始 / リセット」** を押して開始してください。")
        with st.expander("vocab.csv（先頭の数行プレビュー）"):
            st.dataframe(df.head(10))
    else:
        st.info("復習モード：左のチェックで対象を選び、**「▶️ クイズを開始 / リセット」** を押してください。")
    st.stop()

############################
# タイマー（自動リフレッシュ）
############################
def seconds_left():
    if not state.quiz_running or state.finished or state.time_up:
        return 0
    return max(0, int(state.end_time - time.time()))

if state.quiz_running and not state.finished and not state.time_up:
    # 1秒ごとに再描画
    st_autorefresh(interval=1000, key="tick_timer")

    left = seconds_left()
    # 0秒 → タイムアップ
    if left <= 0:
        state.time_up = True

    # 見た目の演出
    klass = "timer-box"
    msg = f"残り時間: {left} 秒"
    if left <= 5 and left > 0:
        klass += " timer-urgent"
        msg += " ⏰ やばいぞ！"
    elif left <= 10 and left > 0:
        klass += " timer-warn"
        msg += " ⏳ 時間がなくなってきている…"

    st.markdown(f'<div class="{klass}">{msg}</div>', unsafe_allow_html=True)
    # 進捗バー（経過ベース）
    if state.time_limit_sec > 0:
        elapsed = state.time_limit_sec - left
        pct = min(1.0, max(0.0, elapsed / state.time_limit_sec))
        st.progress(pct, text=f"残り {left}s / 合計 {state.time_limit_sec}s")

############################
# タイムアップ表示
############################
if state.time_up and not state.finished:
    st.markdown('<div class="timeout-banner">🐰 <b>時間切れ！</b></div>', unsafe_allow_html=True)
    # 集計
    colA, colB, colC = st.columns([1,1,1])
    with colA: st.metric("対象語数", len(state.candidate_indices))
    with colB: st.metric("総出題", state.stats_total)
    with colC:
        acc = (state.stats_correct / max(1, state.stats_total) * 100)
        st.metric("正答率", f"{acc:.1f}%")

    # 復習用ダウンロード（今セッションの誤答 + ⭐）
    wrong_indices_list = [log["idx"] for log in state.logs if not log["correct"]]
    sel = sorted(set(wrong_indices_list).union(state.starred))
    csv_data = export_words(df, sel)
    st.write("---")
    c1, c2 = st.columns([1,1])
    with c1:
        if csv_data:
            st.download_button("🗂️ 誤答/スター語をCSVで保存", data=csv_data,
                               file_name=f"review_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")
    with c2:
        # ログ保存（全体 & 個人）
        if st.button("💾 学習ログを保存"):
            # 全体ログ
            global_log = APP_DIR / "quiz_log.csv"
            exists = global_log.exists()
            pd.DataFrame(state.logs).to_csv(global_log, mode="a", index=False, header=not exists, encoding="utf-8-sig")
            # 個人ログ
            if state.user_id:
                user_log = LOG_DIR / f"{state.user_id}_quiz_log.csv"
                exists_u = user_log.exists()
                pd.DataFrame(state.logs).to_csv(user_log, mode="a", index=False, header=not exists_u, encoding="utf-8-sig")
            st.success("保存しました。")
    st.write("---")
    c3, c4 = st.columns([1,1])
    with c3:
        if st.button("🔁 もう一度 同じ設定で再挑戦"):
            # 同じプールで再スタート
            start_quiz_with_pool(state, state.candidate_indices)
    with c4:
        if st.button("🔁 間違えた語だけで復習を開始（⭐含む）", type="primary"):
            pool = sorted(set(wrong_indices_list).union(state.starred))
            start_quiz_with_pool(state, pool)
    st.stop()

############################
# 進捗表示
############################
if state.candidate_indices:
    mastered_count = len(state.mastered.intersection(set(state.candidate_indices)))
    total_in_range = len(state.candidate_indices)
    pct = mastered_count / total_in_range if total_in_range else 0.0
    st.progress(pct, text=f"進捗: {mastered_count}/{total_in_range}（{int(pct*100)}%）")
    st.caption(f"正答 {state.stats_correct} / 誤答 {state.stats_wrong} / 出題 {state.stats_total}")

############################
# 完了表示
############################
if state.finished:
    st.markdown('<div class="done-banner">🐰 <b>よくやったね！</b></div>', unsafe_allow_html=True)
    wrong_indices_list = [log["idx"] for log in state.logs if not log["correct"]]
    sel = sorted(set(wrong_indices_list).union(state.starred))
    csv_data = export_words(df, sel)
    colA, colB, colC = st.columns([1,1,1])
    with colA: st.metric("対象語数", len(state.candidate_indices))
    with colB: st.metric("総出題", state.stats_total)
    with colC:
        acc = (state.stats_correct / max(1, state.stats_total) * 100)
        st.metric("正答率", f"{acc:.1f}%")
    st.write("---")
    c1, c2 = st.columns([1,1])
    with c1:
        if csv_data:
            st.download_button("🗂️ 誤答/スター語をCSVで保存",
                               data=csv_data,
                               file_name=f"review_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")
    with c2:
        if st.button("💾 学習ログを保存（全体/個人）"):
            global_log = APP_DIR / "quiz_log.csv"
            exists = global_log.exists()
            pd.DataFrame(state.logs).to_csv(global_log, mode="a", index=False, header=not exists, encoding="utf-8-sig")
            if state.user_id:
                user_log = LOG_DIR / f"{state.user_id}_quiz_log.csv"
                exists_u = user_log.exists()
                pd.DataFrame(state.logs).to_csv(user_log, mode="a", index=False, header=not exists_u, encoding="utf-8-sig")
            st.success("保存しました。")
    st.write("---")
    if st.button("🔁 間違えた語だけで復習を開始（⭐含める）", type="primary"):
        pool = sel
        start_quiz_with_pool(state, pool)
    st.stop()

############################
# 問題レンダリング（タイムアップでない時）
############################
idx = state.current_idx
word = df.loc[idx, "word"]
pos = df.loc[idx, "pos"]
meaning = df.loc[idx, "meaning"]
level = df.loc[idx, "level"]

# 出題文
if state.current_direction == "w2m":
    st.markdown(f'<div class="big-word">{word}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="pos-tag">{pos}</span>', unsafe_allow_html=True)
    prompt_sub = f"正しい和訳を選んでください（Lv:{level})"
else:
    st.markdown(f'<div class="big-word">{meaning}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="pos-tag">{pos}</span>', unsafe_allow_html=True)
    prompt_sub = f"正しい英単語を選んでください（Lv:{level})"
st.caption(prompt_sub)

# キーボード選択（1-9 / A-J）
def on_key_change():
    if state.time_up:
        return
    val = st.session_state.get(state.kbd_key, "")
    if not val:
        return
    ch = val.strip().lower()
    pos_sel = None
    if ch.isdigit():
        p = int(ch) - 1
        if 0 <= p < len(state.choices):
            pos_sel = p
    elif 'a' <= ch <= 'z':
        p = ord(ch) - ord('a')
        if 0 <= p < len(state.choices):
            pos_sel = p
    st.session_state[state.kbd_key] = ""  # 入力を即クリア（持ち越し防止）
    if pos_sel is not None:
        process_answer(state, pos_sel)

st.text_input("キーボード入力（1-9 / A-J）", value="", key=state.kbd_key, on_change=on_key_change, label_visibility="collapsed")

# 選択肢描画
def render_choices_interactive():
    for i, ch in enumerate(state.choices):
        label = f"{i+1}. {ch['label']}"
        if st.button(label, key=f"btn_{state.question_id}_{i}", disabled=state.time_up):
            process_answer(state, i)

def render_choices_feedback():
    # 誤答時の1秒フィードバック
    for i, ch in enumerate(state.choices):
        klass = "choice"
        if i == state.correct_choice_pos:
            klass += " correct"
        elif state.last_selected is not None and i == state.last_selected:
            klass += " incorrect"
        st.markdown(f'<div class="{klass}">{i+1}. {ch["label"]}</div>', unsafe_allow_html=True)
    # タイムアップが迫っていても、ここは1秒待ってから進める
    time.sleep(1.0)
    state.pending_feedback = False
    advance_question(state)
    rerun()

if not state.time_up:
    if state.pending_feedback:
        render_choices_feedback()
    else:
        render_choices_interactive()

# 便利機能（スター/辞書リンク/スキップ）
st.write("---")
cA, cB, cC = st.columns([1,1,2])
with cA:
    starred_flag = idx in state.starred
    if st.toggle("⭐ 苦手としてマーク", value=starred_flag, key=f"star_{idx}", disabled=state.time_up):
        state.starred.add(idx)
    else:
        if idx in state.starred:
            state.starred.remove(idx)
with cB:
    st.write("")  # spacing
    if st.button("🔁 スキップ（後で出す）", disabled=state.time_up):
        state.queue.append(idx)
        advance_question(state)
        rerun()
with cC:
    q = word
    weblio = f"https://ejje.weblio.jp/content/{q}"
    cambridge = f"https://dictionary.cambridge.org/dictionary/english/{q}"
    oxford = f"https://www.oxfordlearnersdictionaries.com/definition/english/{q}"
    st.markdown(
        f'<span class="subtle">辞書:</span> Weblio ・ Cambridge ・ Oxford',
        unsafe_allow_html=True
    )
