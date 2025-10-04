# app.py
# -*- coding: utf-8 -*-
import os
import time
import random
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

############################
# スタイル（見やすいUI）
############################
CUSTOM_CSS = """
<style>
main.block-container { max-width: 920px; }

/* 問題の英単語を大きく太く */
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

/* 正解/不正解のハイライト */
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

/* 完了バナー */
.done-banner {
  font-size: 42px;
  font-weight: 900;
  text-align: center;
  padding: 24px 12px;
  border-radius: 16px;
  border: 2px dashed #8ad;
  background: #f5fbff;
}
.subtle {
  color: #666;
  font-size: 14px;
}
kbd {
  padding: 2px 6px;
  border: 1px solid #8882;
  border-bottom-width: 2px;
  border-radius: 6px;
  background: #f8f8f8;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
.mode-badge {
  display:inline-block; padding:4px 10px; border-radius: 999px;
  font-size: 12px; border:1px solid #d0d7de; background:#f6f8fa; color:#333;
}
</style>
"""
st.set_page_config(page_title="TOEIC Vocab Quiz", page_icon="📝", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

############################
# 定数/ユーティリティ
############################
DATA_FILE = "vocab.csv"     # CSV/TSV/セミコロン等を自動判定
LOG_FILE  = "quiz_log.csv" # セッションログの追記先（任意）

def _load_vocab_safely(path: str) -> pd.DataFrame:
    """区切り文字/エンコーディングを自動判定して読み込み、列名を正規化"""
    if not os.path.exists(path):
        st.error(f"ファイルが見つかりません: {path}")
        st.stop()

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
        "meaning": ["meaning", "意味", "和訳", "訳"]
    }
    for target, candidates in col_map_candidates.items():
        for c in df.columns:
            if c in candidates:
                rename_map[c] = target
                break
    df = df.rename(columns=rename_map)

    for col in ["word", "pos", "meaning"]:
        if col not in df.columns:
            st.error(f"vocabファイルに必要な列 '{col}' が見つかりません。列順は 'word, pos, meaning' を推奨します。")
            st.stop()

    for col in ["word", "pos", "meaning"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=["word", "meaning"]).reset_index(drop=True)
    return df


def rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def build_choices(df: pd.DataFrame, correct_idx: int, candidate_indices: list[int], n_choices: int, direction: str):
    pool = [i for i in candidate_indices if i != correct_idx]
    random.shuffle(pool)

    n = min(n_choices - 1, len(pool))
    distractors = pool[:n]

    def meaning_of(i): return df.loc[i, "meaning"]
    def word_of(i): return df.loc[i, "word"]

    if direction == "w2m":
        correct_label = meaning_of(correct_idx)
        seen = {correct_label}
        filtered = []
        for i in distractors:
            m = meaning_of(i)
            if m not in seen:
                filtered.append(i)
                seen.add(m)
        distractors = filtered
        if len(distractors) < n_choices - 1:
            more = [i for i in pool if i not in distractors]
            distractors.extend(more[: (n_choices - 1 - len(distractors))])
        options = [(correct_label, True)] + [(meaning_of(i), False) for i in distractors[:(n_choices - 1)]]
    else:
        correct_label = word_of(correct_idx)
        seen = {correct_label}
        filtered = []
        for i in distractors:
            w = word_of(i)
            if w not in seen:
                filtered.append(i)
                seen.add(w)
        distractors = filtered
        if len(distractors) < n_choices - 1:
            more = [i for i in pool if i not in distractors]
            distractors.extend(more[: (n_choices - 1 - len(distractors))])
        options = [(correct_label, True)] + [(word_of(i), False) for i in distractors[:(n_choices - 1)]]

    random.shuffle(options)
    return [{"label": lab, "is_correct": is_corr} for lab, is_corr in options]


def advance_question(state):
    if not state.queue:
        state.finished = True
        return

    state.current_idx = state.queue.pop(0)
    state.question_id += 1

    if state.direction == "mix":
        direction = random.choice(["w2m", "m2w"])
    else:
        direction = state.direction
    state.current_direction = direction

    available = len(state.candidate_indices)
    n_choices = min(state.n_choices, max(2, available)) if available >= 2 else 2

    state.choices = build_choices(
        state.df,
        state.current_idx,
        state.candidate_indices,
        n_choices,
        state.current_direction
    )

    # キーボード入力を毎問リセット（持ち越し防止）
    state.kbd_key = f"kbd_{state.question_id}"
    st.session_state[state.kbd_key] = ""

    state.pending_feedback = False
    state.last_selected = None
    state.correct_choice_pos = next(i for i, c in enumerate(state.choices) if c["is_correct"])


def process_answer(state, selected_pos: int):
    state.stats_total += 1
    is_correct = state.choices[selected_pos]["is_correct"]
    idx = state.current_idx

    state.attempts[idx] = state.attempts.get(idx, 0) + 1
    state.logs.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "qid": state.question_id,
        "idx": int(idx),
        "word": state.df.loc[idx, "word"],
        "pos": state.df.loc[idx, "pos"],
        "meaning": state.df.loc[idx, "meaning"],
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


def export_wrong_or_starred(df, wrong_indices: list[int], starred: set[int], filename="export.csv"):
    sel = sorted(set(wrong_indices).union(starred))
    if not sel:
        return None
    out = df.loc[sel, ["word", "pos", "meaning"]].copy()
    buffer = io.StringIO()
    out.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


def wrong_indices_from_current_logs(logs: list[dict]) -> set[int]:
    return {int(x["idx"]) for x in logs if not x.get("correct", True)}


def wrong_indices_from_log_file(df: pd.DataFrame, path: str) -> set[int]:
    """quiz_log.csv から誤答語を抽出し、(word,pos,meaning)で現在のDFにマッピング"""
    if not os.path.exists(path):
        return set()
    try:
        logs_df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        # 読めないときはUTF-8無BOMで試す
        logs_df = pd.read_csv(path, encoding="utf-8")

    if not {"word", "pos", "meaning", "correct"}.issubset(set(logs_df.columns)):
        return set()
    logs_wrong = logs_df[logs_df["correct"] == False][["word", "pos", "meaning"]].drop_duplicates()

    df2 = df.reset_index().rename(columns={"index": "idx"})
    merged = pd.merge(
        logs_wrong,
        df2[["idx", "word", "pos", "meaning"]],
        on=["word", "pos", "meaning"],
        how="inner"
    )
    return set(merged["idx"].tolist())


def start_quiz_with_pool(state, pool_indices: list[int]):
    """与えられたインデックスのプールでクイズを初期化して開始"""
    if not pool_indices:
        st.warning("出題対象がありません。範囲や復習対象の設定を確認してください。")
        return
    indices = list(pool_indices)
    random.shuffle(indices)

    state.candidate_indices = indices.copy()
    state.queue = indices.copy()
    state.mastered = set()
    state.attempts = {}
    state.finished = False
    state.logs = []
    state.stats_total = 0
    state.stats_correct = 0
    state.stats_wrong = 0

    advance_question(state)
    rerun()

############################
# 状態初期化
############################
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.df = _load_vocab_safely(DATA_FILE)

    # 設定系
    st.session_state.mode = "normal"  # 'normal' | 'review'
    st.session_state.range_start = 1
    st.session_state.range_end = len(st.session_state.df)
    st.session_state.n_choices = 4
    st.session_state.direction = "w2m"  # 'w2m' | 'm2w' | 'mix'

    # 復習モード設定
    st.session_state.review_include_current_wrong = True
    st.session_state.review_include_starred = True
    st.session_state.review_include_past_log = False

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

state = st.session_state
df = state.df
total_words = len(df)

############################
# サイドバー（設定）
############################
with st.sidebar:
    st.header("⚙️ 設定")
    st.caption(f"データ: **{DATA_FILE}**（{total_words}語）")

    # モード
    mode_label = st.radio(
        "モード",
        options=[("normal", "通常モード（範囲から出題）"), ("review", "復習モード（誤答/スターのみ）")],
        index=0 if state.mode == "normal" else 1,
        format_func=lambda x: x[1],
        key="mode_radio"
    )[0]
    state.mode = mode_label

    if state.mode == "normal":
        st.write("**出題範囲**（行番号・1始まり）")
        c1, c2 = st.columns(2)
        with c1:
            start = st.number_input("開始", min_value=1, max_value=total_words, value=state.range_start, step=1, key="inp_start")
        with c2:
            end = st.number_input("終了", min_value=1, max_value=total_words, value=state.range_end, step=1, key="inp_end")
        if start > end:
            st.warning("開始は終了以下にしてください。")
        else:
            state.range_start, state.range_end = int(start), int(end)
    else:
        st.write("**復習対象の選択**")
        state.review_include_current_wrong = st.checkbox("直近セッションの誤答を含める", value=state.review_include_current_wrong)
        state.review_include_starred = st.checkbox("⭐スター語を含める", value=state.review_include_starred)
        state.review_include_past_log = st.checkbox("過去ログ（quiz_log.csv）の誤答も含める", value=state.review_include_past_log)

        # 参考カウントのプレビュー
        preview_sets = []
        if state.review_include_current_wrong:
            preview_sets.append(("直近誤答", len(wrong_indices_from_current_logs(state.logs))))
        if state.review_include_starred:
            preview_sets.append(("スター", len(state.starred)))
        if state.review_include_past_log:
            preview_sets.append(("過去ログ誤答", len(wrong_indices_from_log_file(df, LOG_FILE))))
        if preview_sets:
            st.caption("候補数プレビュー: " + " / ".join([f"{k}:{v}" for k, v in preview_sets]))

    # 共通設定
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
            s0 = max(1, min(state.range_start, total_words)) - 1
            e0 = max(1, min(state.range_end, total_words)) - 1
            pool = list(range(s0, e0 + 1))
            start_quiz_with_pool(state, pool)
        else:
            pool_set = set()
            if state.review_include_current_wrong:
                pool_set |= wrong_indices_from_current_logs(state.logs)
            if state.review_include_starred:
                pool_set |= set(state.starred)
            if state.review_include_past_log:
                pool_set |= wrong_indices_from_log_file(df, LOG_FILE)
            pool = sorted(pool_set)
            start_quiz_with_pool(state, pool)

############################
# ヘッダー
############################
st.title("📝 TOEIC Vocabulary Quiz")
badge = "通常モード" if state.mode == "normal" else "復習モード（誤答/スター）"
st.markdown(f'<span class="mode-badge">{badge}</span>', unsafe_allow_html=True)
st.caption("正解で即次へ。誤答は正解を1秒ハイライトした後に後回し。全問1回は正答するまで続きます。")

# 未開始時
if not state.queue and state.current_idx is None and not state.finished:
    if state.mode == "normal":
        st.info("左の **「▶️ クイズを開始 / リセット」** を押して開始してください。")
        with st.expander("vocab.sv（先頭の数行プレビュー）"):
            st.dataframe(df.head(10))
    else:
        st.info("復習モード：左のチェックで対象を選び、**「▶️ クイズを開始 / リセット」** を押してください。")
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
    csv_data = export_wrong_or_starred(df, wrong_indices_list, state.starred, filename="review.csv")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        st.metric("語彙数（対象）", len(state.candidate_indices))
    with colB:
        st.metric("総出題", state.stats_total)
    with colC:
        acc = (state.stats_correct / state.stats_total * 100) if state.stats_total else 0
        st.metric("正答率", f"{acc:.1f}%")

    st.write("---")
    c1, c2 = st.columns([1,1])
    with c1:
        if csv_data:
            st.download_button(
                "🗂️ 誤答/スター語をCSVで保存",
                data=csv_data,
                file_name=f"review_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    with c2:
        if st.button("💾 学習ログをファイルに保存（./quiz_log.csv へ追記）"):
            exists = os.path.exists(LOG_FILE)
            pd.DataFrame(state.logs).to_csv(LOG_FILE, mode="a", index=False, header=not exists, encoding="utf-8-sig")
            st.success(f"保存しました: {os.path.abspath(LOG_FILE)}")

    st.write("---")
    # 完了後にワンクリックで復習モードを始められるボタン
    if st.button("🔁 間違えた語だけで復習を開始（⭐含める）", type="primary"):
        pool_set = wrong_indices_from_current_logs(state.logs) | set(state.starred)
        start_quiz_with_pool(state, sorted(pool_set))
    st.stop()

############################
# 問題レンダリング
############################
idx = state.current_idx
word = df.loc[idx, "word"]
pos = df.loc[idx, "pos"]
meaning = df.loc[idx, "meaning"]

if state.current_direction == "w2m":
    st.markdown(f'<div class="big-word">{word}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="pos-tag">{pos}</span>', unsafe_allow_html=True)
    prompt_sub = "正しい和訳を選んでください"
else:
    st.markdown(f'<div class="big-word">{meaning}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="pos-tag">{pos}</span>', unsafe_allow_html=True)
    prompt_sub = "正しい英単語を選んでください"

st.caption(prompt_sub)

# キーボード選択（1-9 / A-J）
def on_key_change():
    val = st.session_state.get(state.kbd_key, "")
    if not val:
        return
    ch = val.strip().lower()
    pos = None
    if ch.isdigit():
        p = int(ch) - 1
        if 0 <= p < len(state.choices):
            pos = p
    elif 'a' <= ch <= 'z':
        p = ord(ch) - ord('a')
        if 0 <= p < len(state.choices):
            pos = p
    st.session_state[state.kbd_key] = ""  # すぐにクリア（持ち越し防止）
    if pos is not None:
        process_answer(state, pos)

st.text_input("キーボード入力（1-9 / A-J）", value="", key=state.kbd_key, on_change=on_key_change, label_visibility="collapsed")

# 選択肢の描画
def render_choices_interactive():
    for i, ch in enumerate(state.choices):
        label = f"{i+1}. {ch['label']}"
        if st.button(label, key=f"btn_{state.question_id}_{i}"):
            process_answer(state, i)

def render_choices_feedback():
    for i, ch in enumerate(state.choices):
        klass = "choice"
        if i == state.correct_choice_pos:
            klass += " correct"
        elif state.last_selected is not None and i == state.last_selected:
            klass += " incorrect"
        st.markdown(f'<div class="{klass}">{i+1}. {ch["label"]}</div>', unsafe_allow_html=True)
    time.sleep(1.0)
    state.pending_feedback = False
    advance_question(state)
    rerun()

if state.pending_feedback:
    render_choices_feedback()
else:
    render_choices_interactive()

# 便利機能（スター/辞書リンク/スキップ）
st.write("---")
cA, cB, cC = st.columns([1,1,2])
with cA:
    starred_flag = idx in state.starred
    if st.toggle("⭐ 苦手としてマーク", value=starred_flag, key=f"star_{idx}"):
        state.starred.add(idx)
    else:
        if idx in state.starred:
            state.starred.remove(idx)
with cB:
    st.write("")  # spacing
    if st.button("🔁 スキップ（後で出す）"):
        state.queue.append(idx)
        advance_question(state)
        rerun()
with cC:
    q = word
    weblio = f"https://ejje.weblio.jp/content/{q}"
    cambridge = f"https://dictionary.cambridge.org/dictionary/english/{q}"
    oxford = f"https://www.oxfordlearnersdictionaries.com/definition/english/{q}"
    st.markdown(
        f'<span class="subtle">辞書:</span> Weblio ・ [Cambridge・ Oxford',
        unsafe_allow_html=True
    )
