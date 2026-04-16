"""
Aplicación Streamlit para predecir la intención de voto en las elecciones generales argentinas de 2023.

La aplicación carga un dataset con variables sociodemográficas y de comportamiento, entrena un
clasificador Naive Bayes Gaussiano y permite a la persona usuaria introducir sus
respuestas mediante un formulario sencillo. Una vez enviada la información, la
aplicación calcula la probabilidad de votar a cada candidato y muestra la clase
predicha junto con una explicación local utilizando SHAP. También se
proporciona un resumen del dataset y una explicación de cómo funciona el
modelo.

Se han optimizado la estructura del código, la gestión de datos y la
presentación de la interfaz gráfica para mejorar la legibilidad y la
experiencia de usuario. Las validaciones se reducen al mínimo necesario para
garantizar coherencia en los valores de entrada.
"""

import os
import numpy as np
import pandas as pd
# Importación condicional de SHAP.  Si la librería no está disponible en el
# entorno, se avisará al usuario y se desactivará la explicación local.  Esto
# permite que la aplicación funcione aunque no se pueda instalar el paquete.
try:
    import shap  # type: ignore
except ImportError:
    shap = None
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.naive_bayes import GaussianNB

st.set_page_config(
    page_title="Clasificador de voto 2023",
    page_icon="🗳️",
    layout="centered",
    initial_sidebar_state="auto",
)

# Diccionarios de mapeo para variables categóricas.  Se definen una vez para
# facilitar su mantenimiento y uso en el formulario y en el preprocesamiento.
P10_OPTIONS = {
    "Voté convencido": 2,
    "Voté al menos peor": 1,
}
P28_3_OPTIONS = {
    "Nunca": 1,
    "A veces": 2,
    "Seguido": 3,
    "Siempre": 4,
}
P35_2_OPTIONS = {
    "En desacuerdo": 1,
    "Ni de acuerdo ni en desacuerdo": 2,
    "De acuerdo": 3,
}

# Nombres descriptivos para cada feature que se mostrarán en el gráfico de SHAP.
FEATURE_DISPLAY_NAMES = {
    "P2": "Edad",
    "P10": "Motivación de voto",
    "P11": "Ideología (1=Izq., 10=Der.)",
    "P28_3": "Asistencia a marchas",
    "P35_2": "Opinión lenguaje inclusivo",
}

# Mapeo de clase a la ruta de imagen correspondiente.  Se utilizan logos de
# partidos o iconos abstractos según disponibilidad.  Las imágenes deben
# encontrarse en la carpeta 'candidate_logos' en el mismo directorio que el
# script.
IMAGE_PATHS = {
    0: os.path.join("candidate_logos", "blank_vote.png"),
    1: os.path.join("candidate_logos", "libertad_avanza.png"),
    2: os.path.join("candidate_logos", "juntos_cambio.png"),
    3: os.path.join("candidate_logos", "union_patria.png"),
    4: os.path.join("candidate_logos", "otros.png"),
}
# Columnas canónicas utilizadas por el modelo.  El orden importa y se usará
# consistentemente en todos los procesos.
FEATURE_COLS = ["P2", "P10", "P11", "P28_3", "P35_2"]

# Mapeo de etiquetas a nombres amigables.  Cambie este diccionario para
# personalizar la salida de la predicción.
CLASS_NAMES = {
    0: "Blanco/No voté",
    1: "Milei",
    2: "Bullrich",
    3: "Massa",
    4: "Otros",
}


@st.cache_data
def load_dataset(path: str = "DATASET.xlsx") -> pd.DataFrame:
    """Carga el dataset desde el disco o solicita al usuario subirlo.

    Se aprovecha la caché de Streamlit para evitar recargar el fichero en
    cada interacción.  Si el archivo no está disponible, se muestra un
    uploader para que la persona usuaria lo proporcione.
    """
    if os.path.exists(path):
        df = pd.read_excel(path)
    else:
        uploaded = st.file_uploader(
            "Subí el archivo de datos (formato .xlsx)", type=["xlsx"], key="dataset_uploader"
        )
        if uploaded is None:
            st.stop()
        df = pd.read_excel(uploaded)
    return df


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas del DataFrame a sus equivalentes canónicos.

    Detecta nombres comunes en español y los mapea a las variables esperadas por
    el modelo.  Esto facilita trabajar con distintas versiones de datasets.
    """
    rename_map = {
        "Edad [P2]": "P2",
        "Fidelidad [P10]": "P10",
        "Autopercepción Ideológica [P11]": "P11",
        "Autopercepcion Ideologica [P11]": "P11",
        "Asistencia a Marchas [P28_3]": "P28_3",
        "Aceptación del Lenguaje Inclusivo [P35_2]": "P35_2",
        "Aceptacion del Lenguaje Inclusivo [P35_2]": "P35_2",
    }
    available_cols = set(df.columns)
    to_rename = {col: rename_map[col] for col in rename_map if col in available_cols}
    if to_rename:
        df = df.rename(columns=to_rename)
    return df


def discover_target(df: pd.DataFrame) -> str:
    """Intenta detectar automáticamente la columna objetivo (target).

    Busca nombres típicos de la variable de intención de voto o columnas
    numéricas con valores en {0,1,2,3,4}.  Si no se encuentra, se lanza una
    excepción.
    """
    candidates = [
        "P9",
        "Intención de Voto [P9]",
        "Intencion de Voto [P9]",
        "voto",
        "Voto",
        "target",
        "Target",
        "y",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    # heurística: única columna no canónica con 5 clases
    for col in df.columns:
        if col not in FEATURE_COLS:
            values = pd.to_numeric(df[col], errors="coerce")
            unique_vals = values.dropna().unique()
            if set(unique_vals).issubset({0, 1, 2, 3, 4}):
                return col
    raise ValueError(
        "No se encontró columna objetivo. Incluye una columna P9 o equivalente con valores 0..4."
    )


@st.cache_data
def train_model(df: pd.DataFrame):
    """Entrena un Naive Bayes Gaussiano a partir de un DataFrame preprocesado.

    Esta función también regresa el conjunto de características (X), el
    vector objetivo (y) y un subconjunto de `X` para utilizar como fondo en
    el cálculo de SHAP.  La caché evita reentrenar cada vez que el usuario
    interactúa con el formulario.
    """
    df = sanitize_columns(df.copy())
    # aseguramos que las columnas necesarias existan
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan las columnas requeridas: {missing}")
    target_col = discover_target(df)
    # conversión a numérico
    for col in FEATURE_COLS + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # eliminar filas sin valor en target
    df = df.dropna(subset=[target_col]).copy()
    # imputación sencilla: mediana para edad y moda para el resto
    df["P2"] = df["P2"].fillna(df["P2"].median())
    for col in ["P10", "P11", "P28_3", "P35_2"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    X = df[FEATURE_COLS].astype(float)
    y = df[target_col].astype(int)
    model = GaussianNB()
    model.fit(X, y)
    bg = X.sample(n=min(len(X), 80), random_state=42)
    return model, X, y, bg, target_col


def build_user_input_form(defaults: dict) -> dict:
    """Construye el formulario de entrada para el usuario.

    Devuelve un diccionario con las respuestas seleccionadas o ingresadas.
    """
    st.subheader("Completá el formulario")
    with st.form("user_input_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            edad = st.number_input(
                "Edad (en 2023)",
                min_value=16,
                max_value=100,
                step=1,
                value=int(defaults.get("P2", 30)),
                help="Introduce tu edad al momento de las elecciones de 2023",
            )
            ideologia = st.slider(
                "En una escala de 1 (izquierda) a 10 (derecha), ¿dónde te ubicas?",
                min_value=1,
                max_value=10,
                value=int(defaults.get("P11", 5)),
            )
            inclusivo = st.selectbox(
                "El lenguaje inclusivo permite que todos se sientan incluidos",
                list(P35_2_OPTIONS.keys()),
                index=list(P35_2_OPTIONS.keys()).index(defaults.get("P35_2", "En desacuerdo")),
            )
        with col2:
            motivacion = st.selectbox(
                "¿Tu candidato te representaba o era el menos malo?",
                list(P10_OPTIONS.keys()),
                index=list(P10_OPTIONS.keys()).index(defaults.get("P10", "Voté convencido")),
            )
            marchas = st.selectbox(
                "Frecuencia de asistencia a marchas o manifestaciones",
                list(P28_3_OPTIONS.keys()),
                index=list(P28_3_OPTIONS.keys()).index(defaults.get("P28_3", "Nunca")),
            )
        enviado = st.form_submit_button("Calcular predicción")
    return {
        "submitted": enviado,
        "P2": edad,
        "P11": ideologia,
        "P35_2": inclusivo,
        "P10": motivacion,
        "P28_3": marchas,
    }


def prepare_features(raw_inputs: dict) -> dict:
    """Convierte las respuestas del usuario en el formato esperado por el modelo.

    Este mapeo traduce las opciones textuales a valores numéricos.  Se asume
    que se han validado previamente los rangos de edad e ideología.
    """
    return {
        "P2": int(raw_inputs["P2"]),
        "P10": P10_OPTIONS[raw_inputs["P10"]],
        "P11": int(raw_inputs["P11"]),
        "P28_3": P28_3_OPTIONS[raw_inputs["P28_3"]],
        "P35_2": P35_2_OPTIONS[raw_inputs["P35_2"]],
    }


def predict_vote(model, features: dict):
    """Realiza la predicción con el modelo entrenado.

    Devuelve el índice de la clase predicha, la probabilidad asociada y la
    distribución de probabilidades completa.
    """
    X_new = pd.DataFrame([features])[FEATURE_COLS].astype(float)
    probs = model.predict_proba(X_new)[0]
    class_idx = int(np.argmax(probs))
    class_prob = float(probs[class_idx])
    return class_idx, class_prob, probs


def explain_prediction(model, bg: pd.DataFrame, features: dict, class_idx: int, class_probs: np.ndarray):
    """Genera una figura de SHAP Waterfall para explicar la predicción.

    Se construye un KernelExplainer cada vez que se llama para evitar
    conflictos de estado interno, lo cual es aceptable dado el tamaño pequeño
    de las muestras.  Devuelve una figura de Matplotlib lista para ser
    mostrada en Streamlit.
    """
    # Si SHAP no está disponible, mostramos un mensaje y devolvemos None
    if shap is None:
        return None
    # Preparar entrada para SHAP
    X_new = np.array([[float(features[col]) for col in FEATURE_COLS]])
    f_proba = lambda X_: model.predict_proba(pd.DataFrame(X_, columns=FEATURE_COLS).astype(float))
    explainer = shap.KernelExplainer(f_proba, bg[FEATURE_COLS].to_numpy(), link="identity")
    shap_values = explainer(X_new)
    # Seleccionar valores para la clase predicha
    if hasattr(shap_values, "values"):
        vals = shap_values.values
        base_values = shap_values.base_values
        if vals.ndim == 2:  # binario
            shap_for_class = vals[0].copy()
            base_value = base_values[0] if np.ndim(base_values) > 0 else float(base_values)
            if class_idx == 0:
                shap_for_class = -shap_for_class
                base_value = 1.0 - base_value
        elif vals.ndim == 3:  # multiclase
            shap_for_class = vals[0, class_idx, :].copy()
            base_value = base_values[0, class_idx] if np.ndim(base_values) == 2 else base_values[class_idx]
        else:
            raise ValueError("Dimensión inesperada de SHAP values")
    else:
        # API legacy
        sv_list = shap_values
        expected = getattr(explainer, "expected_value", None)
        if isinstance(sv_list, list):
            if len(sv_list) == 1:
                shap_for_class = sv_list[0][0].copy()
                base_value = expected[0] if isinstance(expected, (list, np.ndarray)) else float(expected)
                if class_idx == 0:
                    shap_for_class = -shap_for_class
                    base_value = 1.0 - base_value
            else:
                shap_for_class = sv_list[class_idx][0].copy()
                base_value = expected[class_idx] if isinstance(expected, (list, np.ndarray)) else float(expected)
        else:
            # ndarray único
            shap_for_class = np.array(sv_list)[0].copy()
            base_value = expected if np.isscalar(expected) else expected[0]
            if class_idx == 0:
                shap_for_class = -shap_for_class
                base_value = 1.0 - float(base_value)
    # Construir explicación para la cascada
    # Utilizamos nombres descriptivos en lugar de los códigos Pxx
    feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLS]
    shap_exp = shap.Explanation(
        values=shap_for_class,
        base_values=base_value,
        data=X_new[0],
        feature_names=feature_names,
    )
    # Crear figura
    plt.figure(figsize=(7, 5))
    shap.plots.waterfall(shap_exp, show=False, max_display=12)
    plt.title(f"Probabilidad para {CLASS_NAMES.get(class_idx, class_idx)}: {class_probs[class_idx]:.3f}")
    fig = plt.gcf()
    return fig



# Interfaz de usuario

def main() -> None:
    st.title("Clasificador de voto en las elecciones generales de 2023")
    st.caption(
        "Completá el formulario y descubre qué candidato o candidata es más probable que hayas votado."
    )

    # Cargar y preprocesar datos
    df = load_dataset()
    try:
        model, X, y, bg, target_col = train_model(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Mostrar un resumen del dataset en un desplegable
    with st.expander("Ver resumen del dataset"):
        st.write(f"Filas: {len(df)} | Columnas: {len(df.columns)}")
        st.dataframe(df.head())

    # Formulario de entrada
    # Establecer valores por defecto a partir del dataset, si es posible.  Utilizamos la
    # mediana de la edad para inicializar y valores centrales típicos para el resto.
    sanitized = sanitize_columns(df.copy())
    if "P2" in sanitized.columns:
        default_age = int(pd.to_numeric(sanitized["P2"], errors="coerce").median())
    else:
        default_age = 30
    defaults = {
        "P2": default_age,
        "P10": "Voté convencido",
        "P11": 5,
        "P28_3": "Nunca",
        "P35_2": "En desacuerdo",
    }
    user_input = build_user_input_form(defaults)
    if not user_input["submitted"]:
        with st.expander("¿Cómo funciona este modelo?", expanded=False):
            st.write(
                "El modelo aprende patrones de un conjunto de datos de votantes reales mediante un "
                "clasificador de Naive Bayes. Con tus respuestas, calcula la probabilidad de que "
                "hayas votado a cada candidato. La clase con mayor probabilidad es la predicha."
            )
            st.info(
                "Esta herramienta es meramente informativa y podría reflejar sesgos presentes en el "
                "dataset utilizado. No debe emplearse para decisiones que afecten derechos o libertades."
            )
        return
    # Preparar características y realizar predicción
    try:
        features = prepare_features(user_input)
    except KeyError:
        st.error("Hay opciones inválidas en el formulario. Por favor, revisa tus selecciones.")
        return
    class_idx, class_prob, class_probs = predict_vote(model, features)
    pred_name = CLASS_NAMES.get(class_idx, str(class_idx))
    # Mostrar resultado
    st.subheader("Resultado de la predicción")
    st.markdown(
        f"<h2 style='text-align:center;color:#4F8BF9;'>Predicción: {pred_name}</h2>", unsafe_allow_html=True
    )
    st.metric("Probabilidad", f"{class_prob * 100:.1f}%")
    # Mostrar imagen asociada al candidato o categoría
    image_path = IMAGE_PATHS.get(class_idx)
    if image_path and os.path.exists(image_path):
        st.image(image_path, caption=f"Logo asociado a {pred_name}", width=200)
    # Explicación SHAP
    fig = explain_prediction(model, bg, features, class_idx, class_probs)
    if fig is None:
        st.info(
            "La librería SHAP no está disponible en el entorno actual, por lo que no se puede "
            "generar la explicación gráfica de la predicción. Si deseas visualizar el "
            "impacto de cada variable, instala la dependencia `shap` en tu entorno."
        )
    else:
        st.pyplot(fig, use_container_width=True)
        # Información adicional sobre el funcionamiento de la gráfica
        with st.expander("¿Cómo se interpreta el gráfico de SHAP?"):
            st.write(
                "El gráfico en cascada muestra cómo cada respuesta influye en la probabilidad final "
                "de la clase predicha. Las barras a la derecha incrementan la probabilidad mientras "
                "que las barras a la izquierda la reducen. El valor base es la probabilidad media "
                "antes de considerar tus respuestas."
            )
