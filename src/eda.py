import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from utils import (
    BASE_DIR, DATA_DATASET,
    find_latest_model_dir, count_labels
)


def render_eda():
    """Hien thi toan bo phan EDA: Bar chart, Loss chart, Confusion Matrix."""
    st.markdown("---")
    st.subheader("Phan Tich Du Lieu & Mo Hinh (EDA)")

    tab1, tab2, tab3 = st.tabs([
        "Phan bo Nhan Dataset",
        "Loss Chart (Qua trinh Train)",
        "Confusion Matrix"
    ])

    # ==========================================
    # TAB 1: BAR CHART PHAN BO NHAN
    # ==========================================
    with tab1:
        st.markdown("**So luong nhan theo tung loai phuong tien trong tap Train**")
        train_label_dir = os.path.join(DATA_DATASET, "labels", "train")
        val_label_dir   = os.path.join(DATA_DATASET, "labels", "val")

        if not os.path.exists(train_label_dir):
            st.warning("Chua co du lieu train. Hay chay Buoc 2 va 2.5 truoc.")
        else:
            counts_train = count_labels(train_label_dir)
            counts_val   = count_labels(val_label_dir) if os.path.exists(val_label_dir) else {}

            labels_list = list(counts_train.keys())
            vals_train  = [counts_train[l] for l in labels_list]
            vals_val    = [counts_val.get(l, 0) for l in labels_list]

            x     = np.arange(len(labels_list))
            width = 0.35

            fig_eda, ax_eda = plt.subplots(figsize=(9, 4))
            fig_eda.patch.set_facecolor("#0e1117")
            ax_eda.set_facecolor("#0e1117")

            bars1 = ax_eda.bar(x - width/2, vals_train, width,
                               label="Train", color="#4FC3F7", alpha=0.85)
            bars2 = ax_eda.bar(x + width/2, vals_val, width,
                               label="Val",   color="#FFB74D", alpha=0.85)

            for bar in bars1:
                h = bar.get_height()
                if h > 0:
                    ax_eda.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                                str(int(h)), ha="center", va="bottom",
                                color="white", fontsize=9)
            for bar in bars2:
                h = bar.get_height()
                if h > 0:
                    ax_eda.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                                str(int(h)), ha="center", va="bottom",
                                color="white", fontsize=9)

            ax_eda.set_xticks(x)
            ax_eda.set_xticklabels(labels_list, color="white", fontsize=10)
            ax_eda.set_ylabel("So nhan", color="#aaa")
            ax_eda.set_title("Phan bo nhan theo class (Train vs Val)",
                             color="white", fontsize=13, fontweight="bold")
            ax_eda.tick_params(colors="#aaa")
            ax_eda.legend(facecolor="#1a1a2e", labelcolor="white")
            for spine in ax_eda.spines.values():
                spine.set_edgecolor("#333")
            ax_eda.grid(axis="y", linestyle=":", alpha=0.3, color="#555")
            plt.tight_layout()
            st.pyplot(fig_eda)
            plt.close(fig_eda)

            total_train = sum(vals_train)
            st.markdown("**Thong ke phan bo:**")
            df_dist = pd.DataFrame({
                "Class":   labels_list,
                "Train":   vals_train,
                "Val":     vals_val,
                "% Train": [f"{v/total_train*100:.1f}%" if total_train > 0 else "0%"
                            for v in vals_train],
            }).set_index("Class")
            st.dataframe(df_dist, use_container_width=True)

            max_count = max(vals_train) if vals_train else 1
            min_count = min([v for v in vals_train if v > 0], default=1)
            ratio = max_count / min_count if min_count > 0 else 999
            if ratio > 5:
                dominant = labels_list[vals_train.index(max(vals_train))]
                st.warning(
                    f"Mat can bang du lieu: class {dominant!r} chiem uu the gap {ratio:.0f}x "
                    f"so voi class it nhat. Day la nguyen nhan chinh khien model nhan dien kem "
                    f"cac phuong tien xuat hien it trong dataset."
                )
            else:
                st.success("Dataset tuong doi can bang giua cac class.")

    # ==========================================
    # TAB 2: LOSS CHART
    # ==========================================
    with tab2:
        model_dir = find_latest_model_dir(BASE_DIR)
        if model_dir is None:
            st.warning("Chua tim thay thu muc model. Hay huan luyen AI o Buoc 3 truoc.")
        else:
            st.caption(f"Dang doc tu: {os.path.relpath(model_dir, BASE_DIR)}")
            results_png = os.path.join(model_dir, "results.png")
            if os.path.exists(results_png):
                st.markdown("**Bieu do Loss & Metrics qua cac Epoch:**")
                st.image(results_png, use_container_width=True)
                st.markdown(
                    "**Cach doc bieu do:**\n"
                    "- Box/Cls/Dfl loss giam deu: model dang hoc tot\n"
                    "- Val loss <= Train loss: khong bi Overfitting\n"
                    "- mAP50 tang va on dinh: model hoi tu tot"
                )
            else:
                st.warning(f"Khong tim thay results.png trong: {os.path.relpath(model_dir, BASE_DIR)}")

            results_csv = os.path.join(model_dir, "results.csv")
            if os.path.exists(results_csv):
                df_res = pd.read_csv(results_csv)
                df_res.columns = [c.strip() for c in df_res.columns]

                col_l2, col_r2 = st.columns(2)
                with col_l2:
                    st.markdown("**Train Loss vs Val Loss (Classification):**")
                    fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
                    fig_loss.patch.set_facecolor("#0e1117")
                    ax_loss.set_facecolor("#0e1117")
                    if "train/cls_loss" in df_res.columns:
                        ax_loss.plot(df_res["epoch"], df_res["train/cls_loss"],
                                     color="#4FC3F7", linewidth=2, label="Train cls loss")
                    if "val/cls_loss" in df_res.columns:
                        ax_loss.plot(df_res["epoch"], df_res["val/cls_loss"],
                                     color="#EF9A9A", linewidth=2, linestyle="--",
                                     label="Val cls loss")
                    ax_loss.set_xlabel("Epoch", color="#aaa")
                    ax_loss.set_ylabel("Loss", color="#aaa")
                    ax_loss.tick_params(colors="#aaa")
                    ax_loss.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
                    for spine in ax_loss.spines.values():
                        spine.set_edgecolor("#333")
                    ax_loss.grid(linestyle=":", alpha=0.3, color="#555")
                    plt.tight_layout()
                    st.pyplot(fig_loss)
                    plt.close(fig_loss)

                with col_r2:
                    st.markdown("**mAP50 qua cac Epoch:**")
                    fig_map, ax_map = plt.subplots(figsize=(5, 3))
                    fig_map.patch.set_facecolor("#0e1117")
                    ax_map.set_facecolor("#0e1117")
                    if "metrics/mAP50(B)" in df_res.columns:
                        ax_map.plot(df_res["epoch"], df_res["metrics/mAP50(B)"],
                                    color="#81C784", linewidth=2, label="mAP50")
                    if "metrics/mAP50-95(B)" in df_res.columns:
                        ax_map.plot(df_res["epoch"], df_res["metrics/mAP50-95(B)"],
                                    color="#FFB74D", linewidth=2, linestyle="--",
                                    label="mAP50-95")
                    ax_map.set_xlabel("Epoch", color="#aaa")
                    ax_map.set_ylabel("mAP", color="#aaa")
                    ax_map.tick_params(colors="#aaa")
                    ax_map.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
                    for spine in ax_map.spines.values():
                        spine.set_edgecolor("#333")
                    ax_map.grid(linestyle=":", alpha=0.3, color="#555")
                    plt.tight_layout()
                    st.pyplot(fig_map)
                    plt.close(fig_map)

    # ==========================================
    # TAB 3: CONFUSION MATRIX
    # ==========================================
    with tab3:
        model_dir = find_latest_model_dir(BASE_DIR)
        if model_dir is None:
            st.warning("Chua tim thay thu muc model.")
        else:
            st.caption(f"Dang doc tu: {os.path.relpath(model_dir, BASE_DIR)}")
            cm_norm = os.path.join(model_dir, "confusion_matrix_normalized.png")
            cm_raw  = os.path.join(model_dir, "confusion_matrix.png")
            if os.path.exists(cm_norm):
                st.markdown("**Confusion Matrix (Normalized):**")
                st.image(cm_norm, use_container_width=True)
                st.markdown(
                    "**Cach doc:** Duong cheo chinh cang sang cang tot.\n"
                    "O ngoai duong cheo = model dang nhan nham class nay thanh class khac.\n"
                    "Vi du: hang truck - cot car cao = model hay nham truck thanh car."
                )
            elif os.path.exists(cm_raw):
                st.markdown("**Confusion Matrix:**")
                st.image(cm_raw, use_container_width=True)
            else:
                st.warning(
                    f"Khong tim thay confusion_matrix.png trong: "
                    f"{os.path.relpath(model_dir, BASE_DIR)}. "
                    "Dam bao chay huan luyen voi plots=True."
                )