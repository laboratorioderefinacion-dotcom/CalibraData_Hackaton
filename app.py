#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Código APP CalibraData

# Librerías

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


# Configuración de la app

st.set_page_config(page_title="Modelo Isotónico", layout="centered")


# CSS: ocultar el icono de enlace (anchor) en títulos
st.markdown("""
<style>
a.anchor-link { 
    display: none !important; 
}
a[href^="#"] { 
    display: none !important; 
}
</style>
""", unsafe_allow_html=True)


st.title("📊 CalibraData – Modelo Isotónico")

# Subir excel

excel_file = st.file_uploader("📂 Subir archivo Excel", type=["xlsx"])

if excel_file:

    if st.button("🧪 Procesar muestra"):

        with st.spinner("Procesando muestra..."):

            with tempfile.TemporaryDirectory() as tmpdir:

                # Guardar Excel temporal
                excel_path = os.path.join(tmpdir, "datos.xlsx")
                with open(excel_path, "wb") as f:
                    f.write(excel_file.read())

                # Datos extraidos del excel
        
                datos = pd.read_excel(excel_path, header=None)

                equipo = str(datos.iloc[0, 0]).strip()
                variable = str(datos.iloc[1, 0]).strip()

                X = pd.to_numeric(datos.iloc[2:, 0], errors="coerce").to_numpy()
                y = pd.to_numeric(datos.iloc[2:, 1], errors="coerce").astype(int)

                # Modelo Isotónico Pooled

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

                # Gráfico
            
                grafico_path = os.path.join(tmpdir, "grafico.png")

                fig, ax = plt.subplots(figsize=(8, 4))

                # Datos
            
                ax.scatter(
                    X, y,
                    alpha=0.6,
                    label="Observaciones",
                    color="tab:blue"
                )

                # Modelo

                ax.plot(
                    uso_grid, prob_grid,
                    color="red",
                    linewidth=2,
                    label="Modelo isotónico"
                )

                # Línea horizontal P=0.50

                ax.axhline(
                    0.5,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    label="P = 0.50"
                )

                # Línea vertical P50

                if not np.isnan(P50):
                    ax.axvline(
                        P50,
                        color="black",
                        linestyle=":",
                        linewidth=2,
                        label=f"P50 ≈ {P50} {variable}"
                    )

                ax.set_xlabel(f"Uso acumulado ({variable})")
                ax.set_ylabel("Probabilidad de deriva")
                ax.set_title(f"Modelo isotónico – {equipo}")
                ax.legend()

                # ✅ Mostrar en pantalla ANTES del Word

                st.subheader("📈 Gráfico del modelo")
                st.pyplot(fig, clear_figure=False)

                # ✅ Guardar para insertarlo en el Word

                fig.savefig(grafico_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

                # Métricas del Modelo

                with st.expander("📌 Ver Métricas del Modelo"):
                    c1, c2 = st.columns(2)
                    c1.metric("AUC ROC", auc_texto)
                    c2.metric("Brier Score", f"{brier:.3f}")
                    st.caption("AUC: capacidad de discriminar deriva/no deriva. Brier: calibración (más bajo es mejor).")

          
                # Creación de informe

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

            
                # Descargar Informe Word
          
                with open(docx_out, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar Informe Técnico",
                        f.read(),
                        file_name=f"Reporte_{equipo}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

