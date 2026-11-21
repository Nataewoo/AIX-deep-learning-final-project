"""Draw README explanations and a comparison from existing CV results; no training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from paths import CV_DIR, DATA_DIR, FIGURE_DIR, REPO

INK = "#203247"
BLUE = "#315FC7"
TEAL = "#0E7E83"
GRAY = "#64748B"
PALE = "#F3F6FA"


def setup_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", "AppleGothic"):
        if name in installed:
            break
    else:
        raise RuntimeError("Install a Korean font before generating these figures.")
    plt.rcParams.update({"font.family": name, "font.size": 12,
                         "axes.unicode_minus": False, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": GRAY,
                         "ytick.color": INK, "savefig.facecolor": "white"})


def canvas(title, subtitle, size=(12, 5)):
    fig = plt.figure(figsize=size, facecolor="white")
    ax = fig.add_axes([.035, .08, .93, .73])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(.045, .925, title, fontsize=20, weight="bold", va="center")
    fig.text(.045, .855, subtitle, fontsize=12, color=GRAY, va="center")
    return fig, ax


def box(ax, x, y, w, h, label, edge=BLUE, fill=PALE, fontsize=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=0.015",
                              linewidth=1.3, edgecolor=edge, facecolor=fill))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fontsize,
            linespacing=1.6, color=INK)


def arrow(ax, start, end, color=GRAY, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                linewidth=1.4, color=color, **kwargs))


def save(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    border = fig.bbox
    for text in fig.findobj(plt.Text):
        if not text.get_text() or not text.get_visible():
            continue
        rect = text.get_window_extent(renderer)
        if rect.x0 < -1 or rect.y0 < -1 or rect.x1 > border.x1 + 1 or rect.y1 > border.y1 + 1:
            raise ValueError(f"Text extends beyond the image: {text.get_text()}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def timeline(out):
    fig, ax = canvas("잠자리에 든 시각과 잠든 시각은 다릅니다",
                     "시간을 구분하기 위한 가상 예시 · 실제 참가자 관측값이 아닙니다", size=(12, 4.7))
    points = [.07, .30, .55, .94]
    labels = ["00:00\n잠자리에 듦", "00:10\n잠을 청함", "00:30\n잠듦", "08:00\n최종 기상"]
    ax.plot([points[0], points[-1]], [.64, .64], color="#CCD5E0", lw=3)
    for x, label in zip(points, labels):
        ax.scatter(x, .64, s=80, color=INK, zorder=3)
        ax.text(x, .80, label, ha="center", va="center", fontsize=13, linespacing=1.6)
    box(ax, .07, .32, .48, .17, "입면 지연 30분\n침대에 든 시각 → 잠든 시각", edge=BLUE, fill="#EDF2FD")
    box(ax, .57, .32, .37, .17, "수면 구간 7시간 30분\n잠든 시각 → 최종 기상", edge=TEAL, fill="#ECF7F7")
    ax.text(.07, .13, "수면 구간에는 밤중에 깨어 있던 시간도 포함됩니다.", fontsize=12)
    fig.text(.045, .045, "시각 사이의 간격은 설명을 위해 조정했습니다.", fontsize=10, color=GRAY)
    save(fig, out/"10_sleep_timeline.png")


def workflow(out, summary):
    train, test, korea = (summary[key] for key in ("train_melidos_all", "test_melidos", "test_korea"))
    fig, ax = canvas("학습과 평가에서 같은 사람의 기록을 섞지 않았습니다",
                     "학습 대상 안에서 모델을 고른 뒤, 따로 둔 참가자에게 적용합니다", size=(12, 6.4))
    box(ax, .04, .83, .92, .13, "공개 수면일지 · 사람 단위로 학습 대상과 평가 대상 분리", edge=GRAY)
    train_text = f"해외 학습 대상\n{train['participants']}명 · {train['rows']:,}건"
    test_text = f"해외 별도 평가\n{test['participants']}명 · {test['rows']:,}건"
    korea_text = f"국내 외부 평가\n{korea['participants']}명 · {korea['rows']:,}건"
    box(ax, .04, .57, .40, .16, train_text)
    box(ax, .56, .57, .18, .16, test_text, edge=TEAL)
    box(ax, .78, .57, .18, .16, korea_text, edge=TEAL)
    for x in (.24, .65, .87):
        arrow(ax, (x, .83), (x, .74))
    box(ax, .04, .30, .40, .16, "5회 교차검증\n4개 그룹 학습 + 1개 그룹 검증")
    arrow(ax, (.24, .57), (.24, .47))
    box(ax, .04, .03, .40, .16, "선택한 설정으로 전체 학습 대상 재학습\n최종 모델 A · B 저장")
    arrow(ax, (.24, .30), (.24, .20))
    box(ax, .56, .03, .40, .21, "설정을 고정한 모델의 성능·오차 확인\n해외: A·B / 국내: A", edge=TEAL, fill="#ECF7F7")
    arrow(ax, (.65, .57), (.65, .25), color=TEAL)
    arrow(ax, (.87, .57), (.87, .25), color=TEAL)
    arrow(ax, (.44, .11), (.55, .11))
    fig.text(.045, .035, "입력은 해당 밤보다 앞선 기록으로 생성 · 빈 값 대체와 표준화 기준은 학습 자료에서 계산", fontsize=10.5, color=GRAY)
    save(fig, out/"11_project_workflow.png")


def architecture(out, selection):
    fig, ax = canvas("최종 모델 A와 B는 각각 독립적으로 학습합니다",
                     "과거 이력 5개는 공통 · B에는 수면 구간 길이 1개가 더 들어갑니다", size=(12, 6.3))
    for task, y, color in (("A", .60, BLUE), ("B", .16, TEAL)):
        selected = selection[task]["selected"]
        assert selected["model"] == "mlp"
        width = selected["params"]["width"]
        ax.text(.02, y + .30, f"모델 {task}", weight="bold", color=color, fontsize=14)
        inputs = "과거 수면 이력\n5개 숫자" if task == "A" else "같은 이력 5개\n+ 수면 구간 길이"
        output = "예측 입면 지연\n0분 이상" if task == "A" else "예측 수면의 질\n1~5점"
        box(ax, .02, y, .19, .23, inputs, edge=color)
        box(ax, .27, y, .17, .23, "빈 값 대체\n표준화", edge=color)
        box(ax, .50, y, .23, .23, f"MLP 계산 층\n{width} → {width//2} → 1", edge=color)
        box(ax, .79, y, .19, .23, output, edge=color, fill="#ECF7F7" if task == "B" else "#EDF2FD")
        for x1, x2 in ((.21, .265), (.44, .495), (.73, .785)):
            arrow(ax, (x1, y+.115), (x2, y+.115), color=color)
    fig.text(.045, .048, "각 은닉층 뒤 ReLU · 첫 은닉층 뒤 Dropout 10% · 최종 출력은 세 시작 조건의 예측 평균", fontsize=10.5, color=GRAY)
    save(fig, out/"12_model_architecture.png")


def cv_comparison(out, cv):
    kinds = ["baseline", "ridge", "random_forest", "gradient_boosting", "mlp"]
    names = ["기준선", "Ridge", "Random Forest", "Gradient Boosting", "MLP"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(left=.17, right=.965, bottom=.19, top=.77, wspace=.64)
    fig.text(.045, .935, "동일한 참가자 그룹으로 비교한 내부 검증 결과", fontsize=20, weight="bold")
    fig.text(.045, .86, "모델별 최선 설정 · 사람별 평균 오차를 다시 평균 · 작을수록 좋습니다", fontsize=12, color=GRAY)
    for ax, task, unit, color, limit, decimals in zip(axes, ["A","B"], ["분","점"], [BLUE,TEAL], [40,1.06], [2,3]):
        rows = cv[(cv.task==task) & (cv.variant=="full")]
        values = [float(rows[rows.model==kind].macro_mae.min()) for kind in kinds]
        assert int(np.argmin(values)) == kinds.index("mlp")
        bars = ax.barh(names, values, height=.60, color=[color if kind=="mlp" else "#C3CDD9" for kind in kinds])
        ax.bar_label(bars, labels=[f"{value:.{decimals}f}" for value in values], padding=5, fontsize=11)
        ax.set_xlim(0, limit)
        ax.set_xticks([0, 10, 20, 30, 40] if task == "A" else [0, .25, .5, .75, 1])
        ax.invert_yaxis()
        ax.set_title(f"모델 {task} · MAE ({unit})", loc="left", pad=15, fontsize=14, color=INK)
        ax.set_xlabel(f"참가자 평균 절대 오차 ({unit})", labelpad=10)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.045, .055, "모델 선택에 사용한 점수입니다. 별도 참가자의 평가 결과는 본문 4장에서 확인합니다.", fontsize=10.5, color=GRAY)
    save(fig, out/"13_cv_model_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_font()
    summary = json.loads((DATA_DIR/"summary.json").read_text(encoding="utf-8-sig"))
    selection = json.loads((CV_DIR/"selection.json").read_text(encoding="utf-8-sig"))
    timeline(args.output)
    workflow(args.output, summary)
    architecture(args.output, selection)
    cv_comparison(args.output, pd.read_csv(CV_DIR/"cv_summary.csv"))
    print("Created 4 README figures from existing settings and CV results; no retraining.")


if __name__ == "__main__":
    main()
