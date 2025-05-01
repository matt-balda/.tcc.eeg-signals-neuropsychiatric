import pandas as pd
import os

dictionary = pd.DataFrame([
    {
        "variavel": "sex",
        "descricao": "Sexo do paciente",
        "tipo":"qualitativa",
        "subtipo":"nominal"
    },{
        "variavel": "age",
        "descricao": "Idade do paciente",
        "tipo":"quantitativa",
        "subtipo":"discreta"
    },{
        "variavel": "eeg.date",
        "descricao": "Data do registro do EEG",
        "tipo":"qualitativa",
        "subtipo":"ordinal"
    },{
        "variavel": "education",
        "descricao": "Anos de estudo do paciente",
        "tipo":"quantitativa", 
        "subtipo":"discreta"
    },{
        "variavel": "IQ",
        "descricao": "Quociente de inteligência do paciente",
        "tipo":"quantitativa",
        "subtipo":"continua"
    },{
        "variavel": "main.disorder",
        "descricao": "Categoria principal do transtorno do paciente",
        "tipo":"qualitativa",
        "subtipo":"nominal"
    },{
        "variavel": "specific.disorder",
        "descricao": "Categoria específica do transtorno do paciente",
        "tipo":"qualitativa",
        "subtipo":"nominal"
    }
    
    ## 1140 features ((19 canais de PSD + 171 canais de FC) * 6 bandas de frequência)
    ## tipo: quantitativa, subtipo: continua
])

file_path = '../../references/dictionary.csv'

dictionary.to_csv(file_path, index=False)

if os.path.exists(file_path):
    print("O arquivo está no local especificado.")
else:
    print("O arquivo não está no local especificado.")