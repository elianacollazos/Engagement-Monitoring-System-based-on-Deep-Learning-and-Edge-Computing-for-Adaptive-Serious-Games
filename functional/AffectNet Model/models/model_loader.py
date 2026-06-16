import os
import streamlit as st
from tensorflow import keras
from tensorflow.keras.layers import Dense


class CompatibleDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


@st.cache_resource
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))

    emotion_model = keras.models.load_model(
        os.path.join(base_path, "clasificacion_em.h5"),
        custom_objects={"Dense": CompatibleDense},
        compile=False
    )

    va_model = keras.models.load_model(
        os.path.join(base_path, "va_best.h5"),
        custom_objects={"Dense": CompatibleDense},
        compile=False
    )

    return emotion_model, va_model
