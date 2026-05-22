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

    tab1, tab2, tab3, tab4 = st.tabs([
        "Phan bo Nhan Dataset",
        "Loss Chart & Overfitting",
        "So sanh Fine-tuning",
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

                # ---- PHAN TICH OVERFITTING TU DONG ----
                st.markdown("---")
                st.markdown("**Phan tich Overfitting / Underfitting tu dong:**")

                if "train/cls_loss" in df_res.columns and "val/cls_loss" in df_res.columns:
                    # Lay 10 epoch cuoi de danh gia on dinh
                    n_last = min(10, len(df_res))
                    tail   = df_res.tail(n_last)
                    train_last = tail["train/cls_loss"].mean()
                    val_last   = tail["val/cls_loss"].mean()
                    gap        = val_last - train_last
                    gap_ratio  = gap / train_last if train_last > 0 else 0

                    # Tinh xu huong (slope) cua val_loss o 10 epoch cuoi
                    epochs_tail  = tail["epoch"].values.reshape(-1, 1)
                    val_loss_tail = tail["val/cls_loss"].values
                    from sklearn.linear_model import LinearRegression as _LR
                    slope_model = _LR().fit(epochs_tail, val_loss_tail)
                    val_slope   = slope_model.coef_[0]

                    # Ket luan
                    c_diag1, c_diag2, c_diag3 = st.columns(3)
                    c_diag1.metric("Train loss TB (10 epoch cuoi)", f"{train_last:.4f}")
                    c_diag2.metric("Val loss TB (10 epoch cuoi)",   f"{val_last:.4f}")
                    c_diag3.metric("Chenh lech (Val - Train)",
                                   f"{gap:.4f}",
                                   delta=f"{gap_ratio*100:.1f}%",
                                   delta_color="inverse")

                    # Phan tich chi tiet
                    if gap_ratio > 0.3 and val_slope > 0:
                        trang_thai = "OVERFITTING"
                        mo_ta = (
                            "Val loss dang TANG trong khi Train loss giam. "
                            "Model dang hoc thuoc long du lieu train thay vi hoc tong quat. "
                            "Giai phap: them du lieu, dung data augmentation manh hon, "
                            "hoac giam so epoch."
                        )
                        st.error(f"Trang thai: {trang_thai}")
                    elif gap_ratio < -0.05:
                        trang_thai = "UNDERFITTING"
                        mo_ta = (
                            "Val loss thap hon Train loss bat thuong. "
                            "Model chua hoc du hoac dropout/regularization qua manh. "
                            "Giai phap: tang epochs, thu giam freeze layers khi fine-tuning."
                        )
                        st.warning(f"Trang thai: {trang_thai}")
                    elif gap_ratio <= 0.1 and abs(val_slope) < 0.005:
                        trang_thai = "HOI TU TOT"
                        mo_ta = (
                            "Train loss va Val loss gan nhau va on dinh. "
                            "Model dang tong quat hoa tot tren du lieu chua thay. "
                            "Day la trang thai ly tuong."
                        )
                        st.success(f"Trang thai: {trang_thai}")
                    else:
                        trang_thai = "DANG HOC (chua hoi tu)"
                        mo_ta = (
                            "Val loss van dang giam, model chua dat diem toi uu. "
                            "Co the tiep tuc fine-tuning them de cai thien."
                        )
                        st.info(f"Trang thai: {trang_thai}")

                    st.info(mo_ta)

                    # Bieu do gap giua train va val loss
                    if len(df_res) >= 5:
                        fig_gap, ax_gap = plt.subplots(figsize=(9, 3))
                        fig_gap.patch.set_facecolor("#0e1117")
                        ax_gap.set_facecolor("#0e1117")
                        gap_series = df_res["val/cls_loss"] - df_res["train/cls_loss"]
                        colors_gap = ["#EF9A9A" if g > 0.1 else "#81C784"
                                      for g in gap_series]
                        ax_gap.bar(df_res["epoch"], gap_series,
                                   color=colors_gap, alpha=0.7, width=0.8)
                        ax_gap.axhline(y=0,   color="white",   linestyle="-",  linewidth=1)
                        ax_gap.axhline(y=0.1, color="#EF9A9A", linestyle="--",
                                       linewidth=1, label="Nguong Overfitting (0.1)")
                        ax_gap.set_xlabel("Epoch", color="#aaa")
                        ax_gap.set_ylabel("Val Loss - Train Loss", color="#aaa")
                        ax_gap.set_title(
                            "Khoang cach Val Loss - Train Loss (> 0: nguy co Overfitting)",
                            color="white", fontsize=11, fontweight="bold"
                        )
                        ax_gap.tick_params(colors="#aaa")
                        ax_gap.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
                        for spine in ax_gap.spines.values():
                            spine.set_edgecolor("#333")
                        ax_gap.grid(axis="y", linestyle=":", alpha=0.3, color="#555")
                        plt.tight_layout()
                        st.pyplot(fig_gap)
                        plt.close(fig_gap)
                        st.caption(
                            "Thanh DO: Val loss cao hon Train loss qua 0.1 (nguy co Overfitting). "
                            "Thanh XANH: Model dang tong quat hoa tot."
                        )

    # ==========================================
    # TAB 3: SO SANH TRUOC/SAU FINE-TUNING
    # ==========================================
    with tab3:
        import glob as _glob
        import re as _re

        st.markdown("**So sanh hieu suat qua cac lan Fine-tuning**")
        st.caption(
            "Moi lan nhan nut Huan luyen, YOLO tao ra 1 thu muc traffic_model* moi. "
            "Bang nay tong hop ket qua cua tat ca cac lan train de thay ro su cai thien."
        )

        # Quet tat ca thu muc traffic_model*
        search_dirs = [
            os.path.join(BASE_DIR, "runs", "detect", "results"),
            os.path.join(BASE_DIR, "results"),
        ]
        all_model_dirs = []
        for sd in search_dirs:
            if os.path.exists(sd):
                pattern = os.path.join(sd, "traffic_model*")
                all_model_dirs.extend(
                    [p for p in _glob.glob(pattern) if os.path.isdir(p)]
                )

        def _get_num(path):
            m = _re.search(r"traffic_model-?(\d*)", os.path.basename(path))
            return int(m.group(1)) if m and m.group(1) else 0

        all_model_dirs.sort(key=_get_num)

        if not all_model_dirs:
            st.warning("Chua tim thay thu muc model nao. Hay huan luyen AI o Buoc 3 truoc.")
        else:
            rows_ft = []
            for mdir in all_model_dirs:
                csv_path = os.path.join(mdir, "results.csv")
                if not os.path.exists(csv_path):
                    continue
                df_m = pd.read_csv(csv_path)
                df_m.columns = [c.strip() for c in df_m.columns]
                num      = _get_num(mdir)
                name     = "Train goc" if num == 0 else f"Fine-tune lan {num}"
                n_epochs = len(df_m)

                map_col  = "metrics/mAP50(B)"
                prec_col = "metrics/precision(B)"
                rec_col  = "metrics/recall(B)"
                map9595  = "metrics/mAP50-95(B)"

                best_map   = df_m[map_col].max()   if map_col  in df_m.columns else None
                best_prec  = df_m[prec_col].max()  if prec_col in df_m.columns else None
                best_rec   = df_m[rec_col].max()   if rec_col  in df_m.columns else None
                best_map95 = df_m[map9595].max()   if map9595  in df_m.columns else None

                rows_ft.append({
                    "Lan train":      name,
                    "So epoch":       n_epochs,
                    "mAP50 (best)":   round(best_map,   4) if best_map   is not None else "-",
                    "mAP50-95":       round(best_map95, 4) if best_map95 is not None else "-",
                    "Precision":      round(best_prec,  4) if best_prec  is not None else "-",
                    "Recall":         round(best_rec,   4) if best_rec   is not None else "-",
                })

            if rows_ft:
                df_ft = pd.DataFrame(rows_ft).set_index("Lan train")
                st.dataframe(df_ft, use_container_width=True)

                # Ve bieu do mAP50 qua cac lan train
                map_vals = [r["mAP50 (best)"] for r in rows_ft
                            if r["mAP50 (best)"] != "-"]
                lan_vals = [r["Lan train"] for r in rows_ft
                            if r["mAP50 (best)"] != "-"]

                if len(map_vals) >= 2:
                    fig_ft, ax_ft = plt.subplots(figsize=(9, 4))
                    fig_ft.patch.set_facecolor("#0e1117")
                    ax_ft.set_facecolor("#0e1117")

                    colors_ft = ["#81C784" if v == max(map_vals) else "#4FC3F7"
                                 for v in map_vals]
                    bars_ft = ax_ft.bar(lan_vals, map_vals,
                                        color=colors_ft, alpha=0.85, width=0.5)

                    # Hien so tren moi thanh
                    for bar, val in zip(bars_ft, map_vals):
                        ax_ft.text(
                            bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.005,
                            f"{val:.4f}",
                            ha="center", va="bottom",
                            color="white", fontsize=9, fontweight="bold"
                        )

                    # Duong xu huong
                    x_idx = list(range(len(map_vals)))
                    ax_ft.plot(x_idx, map_vals, color="#FFB74D",
                               linewidth=2, marker="o", markersize=5,
                               linestyle="--", label="Xu huong")

                    ax_ft.set_ylabel("mAP50 (best)", color="#aaa")
                    ax_ft.set_title(
                        "Tien trinh cai thien mAP50 qua cac lan Fine-tuning",
                        color="white", fontsize=13, fontweight="bold"
                    )
                    ax_ft.tick_params(colors="#aaa", axis="y")
                    ax_ft.tick_params(colors="white", axis="x", labelsize=9)
                    plt.xticks(rotation=15, ha="right")
                    ax_ft.set_ylim(0, min(1.0, max(map_vals) + 0.1))
                    ax_ft.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
                    for spine in ax_ft.spines.values():
                        spine.set_edgecolor("#333")
                    ax_ft.grid(axis="y", linestyle=":", alpha=0.3, color="#555")
                    plt.tight_layout()
                    st.pyplot(fig_ft)
                    plt.close(fig_ft)

                    # Nhan xet tu dong
                    best_idx  = map_vals.index(max(map_vals))
                    first_val = map_vals[0]
                    best_val  = max(map_vals)
                    improve   = (best_val - first_val) / first_val * 100 if first_val > 0 else 0

                    st.success(
                        f"Fine-tuning da cai thien mAP50 tu {first_val:.4f} "
                        f"len {best_val:.4f} (+{improve:.1f}%). "
                        f"Ket qua tot nhat: {lan_vals[best_idx]}."
                    )

                    if best_val >= 0.8:
                        st.info("mAP50 >= 0.8: Model dat nguong TOT cho ung dung thuc te.")
                    elif best_val >= 0.5:
                        st.info("mAP50 0.5-0.8: Model TAM DUOC, nen tiep tuc fine-tuning.")
                    else:
                        st.warning("mAP50 < 0.5: Model YEU, can them du lieu chat luong hon.")
            else:
                st.warning("Cac thu muc model khong chua file results.csv.")

    # ==========================================
    # TAB 4: CONFUSION MATRIX & PHAN TICH PER CLASS
    # ==========================================
    with tab4:
        model_dir = find_latest_model_dir(BASE_DIR)
        if model_dir is None:
            st.warning("Chua tim thay thu muc model.")
        else:
            st.caption(f"Dang doc tu: {os.path.relpath(model_dir, BASE_DIR)}")

            # --- 3A: Hien anh Confusion Matrix ---
            cm_norm = os.path.join(model_dir, "confusion_matrix_normalized.png")
            cm_raw  = os.path.join(model_dir, "confusion_matrix.png")
            if os.path.exists(cm_norm):
                st.markdown("**Confusion Matrix (Normalized):**")
                st.image(cm_norm, use_container_width=True)
                st.markdown(
                    "**Cach doc:** Duong cheo chinh cang sang cang tot. "
                    "O ngoai duong cheo = model dang nhan nham class nay thanh class khac. "
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

            st.markdown("---")

            # --- 3B: Phan tich per-class tu results.csv ---
            results_csv = os.path.join(model_dir, "results.csv")
            if not os.path.exists(results_csv):
                st.info("Khong tim thay results.csv de phan tich per-class.")
            else:
                df_res = pd.read_csv(results_csv)
                df_res.columns = [c.strip() for c in df_res.columns]

                st.markdown("**Phan tich Precision & Recall theo tung Epoch:**")
                st.caption(
                    "Precision: Khi model noi day la loai X, no dung bao nhieu %.\n"
                    "Recall: Trong tat ca xe loai X thuc te, model tim thay bao nhieu %."
                )

                # Ve bieu do Precision va Recall toan bo
                fig_pr, ax_pr = plt.subplots(figsize=(9, 4))
                fig_pr.patch.set_facecolor("#0e1117")
                ax_pr.set_facecolor("#0e1117")

                prec_col = "metrics/precision(B)"
                rec_col  = "metrics/recall(B)"

                if prec_col in df_res.columns:
                    ax_pr.plot(df_res["epoch"], df_res[prec_col],
                               color="#4FC3F7", linewidth=2, label="Precision")
                if rec_col in df_res.columns:
                    ax_pr.plot(df_res["epoch"], df_res[rec_col],
                               color="#FFB74D", linewidth=2, linestyle="--", label="Recall")

                # Danh dau epoch tot nhat (mAP50 cao nhat)
                map_col = "metrics/mAP50(B)"
                if map_col in df_res.columns:
                    best_epoch = df_res.loc[df_res[map_col].idxmax(), "epoch"]
                    ax_pr.axvline(x=best_epoch, color="#81C784",
                                  linestyle=":", linewidth=1.5, label=f"Best epoch ({int(best_epoch)})")

                ax_pr.set_xlabel("Epoch", color="#aaa")
                ax_pr.set_ylabel("Score", color="#aaa")
                ax_pr.set_ylim(0, 1.05)
                ax_pr.set_title("Precision & Recall qua cac Epoch",
                                color="white", fontsize=12, fontweight="bold")
                ax_pr.tick_params(colors="#aaa")
                ax_pr.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
                for spine in ax_pr.spines.values():
                    spine.set_edgecolor("#333")
                ax_pr.grid(linestyle=":", alpha=0.3, color="#555")
                plt.tight_layout()
                st.pyplot(fig_pr)
                plt.close(fig_pr)

                # Bang tong ket epoch tot nhat
                if map_col in df_res.columns and prec_col in df_res.columns and rec_col in df_res.columns:
                    best_row = df_res.loc[df_res[map_col].idxmax()]
                    st.markdown("**Chi so tai Epoch tot nhat (mAP50 cao nhat):**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Epoch",     int(best_row["epoch"]))
                    c2.metric("mAP50",     f"{best_row[map_col]:.4f}")
                    c3.metric("Precision", f"{best_row[prec_col]:.4f}")
                    c4.metric("Recall",    f"{best_row[rec_col]:.4f}")

                    # Nhan xet tu dong
                    prec_val = best_row[prec_col]
                    rec_val  = best_row[rec_col]
                    map_val  = best_row[map_col]

                    nhan_xet = []
                    if map_val >= 0.8:
                        nhan_xet.append("mAP50 >= 0.8: Model dat nguong TOT, du dung thuc te.")
                    elif map_val >= 0.5:
                        nhan_xet.append("mAP50 0.5-0.8: Model TAM DUOC, nen train them.")
                    else:
                        nhan_xet.append("mAP50 < 0.5: Model YEU, can them du lieu va train lai.")

                    if prec_val > rec_val + 0.1:
                        nhan_xet.append(
                            f"Precision ({prec_val:.2f}) cao hon Recall ({rec_val:.2f}) nhieu: "
                            "Model it nham nhung hay bo sot xe. Nen ha conf threshold."
                        )
                    elif rec_val > prec_val + 0.1:
                        nhan_xet.append(
                            f"Recall ({rec_val:.2f}) cao hon Precision ({prec_val:.2f}) nhieu: "
                            "Model tim duoc nhieu xe nhung hay nham. Nen tang conf threshold."
                        )
                    else:
                        nhan_xet.append(
                            f"Precision ({prec_val:.2f}) va Recall ({rec_val:.2f}) can bang tot."
                        )

                    st.info("\n".join([f"- {n}" for n in nhan_xet]))