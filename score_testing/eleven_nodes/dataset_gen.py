import pandas as pd
import numpy as np

n = 10000
np.random.seed(42)

# 1️⃣ Variables básicas
edad = np.random.choice(['joven', 'adulto', 'mayor'], n, p=[0.4, 0.4, 0.2])
genero_usuario = np.random.choice(['hombre', 'mujer'], n, p=[0.5, 0.5])
hora = np.random.choice(['mañana', 'tarde', 'noche'], n, p=[0.3, 0.4, 0.3])
dia = np.random.choice(['laboral', 'fin_semana'], n, p=[0.7, 0.3])
duracion = np.random.choice(['corta', 'media', 'larga'], n, p=[0.4, 0.4, 0.2])

# 2️⃣ Interés previo (depende de edad y género)
interes_previo = []
for e, g in zip(edad, genero_usuario):
    if e == 'joven' and g == 'hombre':
        interes = np.random.choice(['entretenimiento', 'película', 'noticias'], p=[0.5, 0.3, 0.2])
    elif e == 'mayor' and g == 'mujer':
        interes = np.random.choice(['película', 'noticias', 'entretenimiento'], p=[0.5, 0.4, 0.1])
    elif e == 'adulto':
        interes = np.random.choice(['noticias', 'película', 'entretenimiento'], p=[0.4, 0.4, 0.2])
    else:
        interes = np.random.choice(['entretenimiento', 'noticias', 'película'], p=[0.4, 0.4, 0.2])
    interes_previo.append(interes)

# 3️⃣ Tipo de emisión (depende de hora y día)
tipo_emision = []
for h, d in zip(hora, dia):
    if d == 'fin_semana' and h == 'noche':
        tipo = np.random.choice(['bajo_demanda', 'diferido', 'directo'], p=[0.6, 0.2, 0.2])
    elif h == 'mañana':
        tipo = np.random.choice(['directo', 'diferido'], p=[0.7, 0.3])
    else:
        tipo = np.random.choice(['directo', 'bajo_demanda', 'diferido'], p=[0.4, 0.4, 0.2])
    tipo_emision.append(tipo)

# 4️⃣ Género del programa (influenciado por edad, hora, día, interés previo, duración y tipo_emision)
generos = []
for e, h, d, i, dur, te in zip(edad, hora, dia, interes_previo, duracion, tipo_emision):
    if i == 'noticias':
        g = np.random.choice(['noticias', 'entretenimiento'], p=[0.7, 0.3])
    elif i == 'película':
        g = np.random.choice(['película', 'entretenimiento'], p=[0.7, 0.3])
    elif te == 'directo':
        g = np.random.choice(['noticias', 'entretenimiento'], p=[0.6, 0.4])
    elif te == 'bajo_demanda':
        g = np.random.choice(['película', 'entretenimiento'], p=[0.6, 0.4])
    elif dur == 'corta':
        g = np.random.choice(['noticias', 'entretenimiento'], p=[0.4, 0.6])
    elif dur == 'larga':
        g = np.random.choice(['película', 'entretenimiento'], p=[0.7, 0.3])
    else:
        g = np.random.choice(['noticias', 'entretenimiento', 'película'], p=[0.4, 0.4, 0.2])
    generos.append(g)

# 5️⃣ Popularidad (depende del tipo de programa)
popularidad = []
for g in generos:
    if g == 'noticias':
        p = np.random.choice(['alta', 'media', 'baja'], p=[0.4, 0.4, 0.2])
    elif g == 'entretenimiento':
        p = np.random.choice(['alta', 'media', 'baja'], p=[0.6, 0.3, 0.1])
    else:  # película
        p = np.random.choice(['alta', 'media', 'baja'], p=[0.5, 0.3, 0.2])
    popularidad.append(p)

# 6️⃣ Satisfacción depende del género, interés previo, popularidad y duración
satisf = []
for g, i, p, dur in zip(generos, interes_previo, popularidad, duracion):
    if g == i:
        base = [0.7, 0.2, 0.1]
    elif p == 'alta':
        base = [0.6, 0.3, 0.1]
    elif p == 'baja':
        base = [0.3, 0.4, 0.3]
    elif dur == 'larga' and g == 'película':
        base = [0.6, 0.3, 0.1]
    else:
        base = [0.4, 0.4, 0.2]
    s = np.random.choice(['alta', 'media', 'baja'], p=base)
    satisf.append(s)

# 7️⃣ Recomendado según satisfacción y popularidad
recom = []
for s, p in zip(satisf, popularidad):
    if s == 'alta' and p in ['media', 'alta']:
        r = np.random.choice(['sí', 'no'], p=[0.9, 0.1])
    elif s == 'media':
        r = np.random.choice(['sí', 'no'], p=[0.6, 0.4])
    else:
        r = np.random.choice(['sí', 'no'], p=[0.3, 0.7])
    recom.append(r)

# 8️⃣ Crear DataFrame final
df = pd.DataFrame({
    'GeneroUsuario': genero_usuario,
    'EdadUsuario': edad,
    'Hora': hora,
    'DiaSemana': dia,
    'DuracionPrograma': duracion,
    'TipoEmision': tipo_emision,
    'InteresPrevio': interes_previo,
    'GeneroPrograma': generos,
    'PopularidadPrograma': popularidad,
    'Satisfaccion': satisf,
    'Recomendado': recom
})

df.to_csv("c" \
"tv_bn_dataset.csv", index=False)
print("✅ Dataset generado con éxito")
print(df.head())
