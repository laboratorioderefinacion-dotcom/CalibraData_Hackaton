#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from mailmerge import MailMerge
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# ==============================
# CONFIG
# ==============================

st.set_page_config(page_title="Modelo Isotónico", layout="centered")

st.title("📊 CalibraData – Modelo Isotónico")

# ==============================
# INPUT
# ==============================
excel_file = st.file_uploader("📂 Subir archivo Excel", type=["xlsx"])

# ==============================
# PROCESO
# ==============================
if excel_file:

    if st.button("🚀 Generar informe"):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Guardar Excel temporal
            excel_path = os.path.join(tmpdir, "datos.xlsx")
            with open(excel_path, "wb") as f:
                f.write(excel_file.read())

            # ==============================
            # CARGA DE DATOS
            # ==============================
            datos = pd.read_excel(excel_path, header=None)

            equipo = str(datos.iloc[0, 0]).strip()
            variable = str(datos.iloc[1, 0]).strip()

            X = pd.to_numeric(datos.iloc[2:, 0], errors="coerce").to_numpy()
            y = pd.to_numeric(datos.iloc[2:, 1], errors="coerce").astype(int)

            # ==============================
            # MODELO
            # ==============================
            modelo = IsotonicRegression(increasing=True, out_of_bounds="clip")
            modelo.fit(X, y)

            uso_grid = np.linspace(X.min(), X.max(), 5000)
            prob_grid = modelo.predict(uso_grid)

            idx = np.where(prob_grid >= 0.5)[0]

            if len(idx) == 0:
                P50 = np.nan
                interpretacion = (
                    "Dentro del rango observado de uso acumulado, la probabilidad estimada de deriva "
                    "no alcanza el 50%, por lo que no se identifica un umbral P50 en los datos disponibles."
            )
            else:
                P50 = int(round(uso_grid[idx[0]]))
                interpretacion = (
                    f"El valor de P50 indica que, a partir de aproximadamente {P50} {variable} acumulados, "
                    "la probabilidad estimada de deriva alcanza el 50%, lo que sugiere un umbral operativo relevante "
                    "para la toma de decisiones respecto a la frecuencia de verificación o calibración."
            )

            y_prob = modelo.predict(X)

            if len(np.unique(y)) > 1:
                auc = roc_auc_score(y, y_prob)
                auc_texto = f"{auc:.3f}"
            else:
                auc = None
                auc_texto = "N/A"

            brier = brier_score_loss(y, y_prob)

            # ==============================
            # GRÁFICO
            # ==============================
            
            grafico_path = os.path.join(tmpdir, "grafico.png")

            plt.figure(figsize=(8, 4))

            # Datos
            plt.scatter(
                X,
                y,
                alpha=0.6,
                label="Observaciones",
                color="tab:blue"
            )

            # Modelo
            plt.plot(
                uso_grid,
                prob_grid,
                color="red",
                linewidth=2,
                label="Modelo isotónico"
            )

            # Línea horizontal P=0.50
            plt.axhline(
                0.5,
                color="gray",
                linestyle="--",
                linewidth=1,
                label="P = 0.50"
            )

            # Línea vertical P50
            if not np.isnan(P50):
                plt.axvline(
                    P50,
                    color="black",
                    linestyle=":",
                    linewidth=2,
                    label=f"P50 ≈ {P50}"
                )

            # Etiquetas
            plt.xlabel(f"Uso acumulado ({variable})")
            plt.ylabel("Probabilidad de deriva")
            plt.title(f"Modelo isotónico – {equipo}")

            # Leyenda
            plt.legend()

            # Guardar
            plt.savefig(grafico_path, dpi=200, bbox_inches="tight")
            plt.close()

            # ==============================
            # WORD
            # ==============================
            plantilla = "Formulario.docx"

            doc = MailMerge(plantilla)

            doc.merge(
                equipo=equipo,
                variable=variable,
                AUC_ROC=auc_texto,
                Brier_Score=f"{brier:.3f}",
                P50="No alcanzado" if np.isnan(P50) else str(P50),
                interpretacion_tecnica=interpretacion
            )

            docx_out = os.path.join(tmpdir, "reporte.docx")
            doc.write(docx_out)

            # Insertar gráfico en Word
            d = Document(docx_out)

            for p in d.paragraphs:
                if "Gráfico" in p.text:
                    p.text = ""
                    run = p.add_run()
                    run.add_picture(grafico_path, width=Inches(4))
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

            d.save(docx_out)

            # ==============================
            # DESCARGA
            # ==============================
            with open(docx_out, "rb") as f:
                st.download_button(
                    "⬇️ Descargar informe Word",
                    f.read(),
                    file_name=f"Reporte_{equipo}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

            st.success("✅ Informe generado correctamente")

