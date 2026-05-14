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


# ---------------------------
# Funciones semáforo (color del número)
# ---------------------------
def color_auc(auc_value):
    """
    AUC ROC (discriminación)
    🟢 >= 0.80
    🟡 0.70 - 0.79
    🔴 < 0.70
    """
    if auc_value is None or (isinstance(auc_value, float) and np.isnan(auc_value)):
        return "#6b7280", "No disponible"  # gris

    if auc_value >= 0.80:
        return "#16a34a", "Excelente"         # verde
    elif auc_value >= 0.70:
        return "#f59e0b", "Aceptable"     # amarillo
    else:
        return "#dc2626", "Revisar"       # rojo


def color_brier(brier_value):
    """
    Brier Score (calibración)
    🟢 <= 0.20
    🟡 0.20 - 0.30
    🔴 > 0.30
    """
    if brier_value is None or (isinstance(brier_value, float) and np.isnan(brier_value)):
        return "#6b7280", "No disponible"  # gris

    if brier_value <= 0.20:
        return "#16a34a", "Excelente"      # verde
    elif brier_value <= 0.30:
        return "#f59e0b", "Aceptable"      # amarillo
    else:
        return "#dc2626", "Revisar"        # rojo


# ---------------------------
# Configuración de la app
# ---------------------------
st.set_page_config(page_title="Modelo Regresión Isotónica", layout="centered")


# ---------------------------
# CSS: ocultar el icono de enlace (anchor) + estilos tarjetas métricas con color
# ---------------------------
st.markdown("""
<style>
a.anchor-link { display: none !important; }
a[href^="#"] { display: none !important; }

.block-container { padding-top: 2rem; }

/* “Tarjetas” para métricas con número coloreado */
.metric-box{
  padding: 0.8rem 0.9rem;
  border-radius: 18px;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

.metric-label{
  font-size: 18px;
  font-weight: 700;
  opacity: 0.85;
}

.metric-value{
  font-size: 40px;
  font-weight: 900;
  line-height: 1.1;
  margin-top: .15rem;
}

.metric-sub{
  margin-top: .25rem;
  font-size: 0.9rem;
  opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)


st.title("📊 CalibraData")

# Subir excel
excel_file = st.file_uploader("📂 Subir archivo Excel", type=["xlsx"])

if excel_file:

    if st.button("🧪 Procesar muestra"):

        with tempfile.TemporaryDirectory() as tmpdir:

            # Guardar Excel temporal
            excel_path = os.path.join(tmpdir, "datos.xlsx")
            with open(excel_path, "wb") as f:
                f.write(excel_file.read())

            # Datos extraídos del excel
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

            # Métricas
            if len(np.unique(y)) > 1:
                auc = roc_auc_score(y, y_prob)
                auc_texto = f"{auc:.3f}"
                auc_num = float(auc)
            else:
                auc = None
                auc_num = None
                auc_texto = "N/A"

            brier = brier_score_loss(y, y_prob)

            # ---------------------------
            # Gráfico
            # ---------------------------
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
                label="Modelo Regresión Isotónica"
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
            ax.set_title(f"Modelo Regresión Isotónica – {equipo}")
            ax.legend()

            # ✅ Mostrar en pantalla ANTES del Word
            st.subheader("📈 Gráfico del modelo")
            st.pyplot(fig, clear_figure=False)

            # ✅ Guardar para insertarlo en el Word
            fig.savefig(grafico_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            # ---------------------------
            # Métricas del Modelo (números en color)
            # ---------------------------

            with st.expander("📌 Ver métricas del modelo"):

                c1, c2 = st.columns(2)

                # Colores según semáforo
                auc_col, auc_estado = color_auc(auc_num)
                br_col, br_estado = color_brier(brier)

                # Variante elegante (compacta)
                criterio_auc = "🟢≥0.80 | 🟡0.70–0.79 | 🔴<0.70"
                criterio_brier = "🟢<0.10 | 🟡0.10–0.20 | 🔴>0.20"

                # AUC
                with c1:
                    st.markdown(f"""
                    <div class="metric-box">
                      <div class="metric-label">AUC ROC</div>
                      <div class="metric-value" style="color:{auc_col};">{auc_texto}</div>
                      <div class="metric-sub">{auc_estado}</div>
                      <div class="metric-sub" style="opacity:.68; font-size:0.85rem; margin-top:.30rem;">
                        <b>Criterio:</b> {criterio_auc}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Brier
                with c2:
                    st.markdown(f"""
                    <div class="metric-box">
                      <div class="metric-label">Brier Score</div>
                      <div class="metric-value" style="color:{br_col};">{brier:.3f}</div>
                      <div class="metric-sub">{br_estado}</div>
                      <div class="metric-sub" style="opacity:.68; font-size:0.85rem; margin-top:.30rem;">
                        <b>Criterio:</b> {criterio_brier}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    "*El AUC ROC refleja la capacidad del modelo para identificar correctamente casos con deriva.  \n"
                    "Mientras que el Brier Score evalúa la confiabilidad de las probabilidades estimadas.  \n"
                    "Valores altos de AUC y bajos de Brier indican un desempeño adecuado del modelo.*"
                )
       
            # ---------------------------
            # Creación de informe
            # ---------------------------
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
                    file_name=f"Informe {equipo}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

