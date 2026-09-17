from fastapi import FastAPI, UploadFile

modelo = {}

app = FastAPI()

@app.get("/")
def raiz():
    return{"messsagem": "math-ocr API"}

@app.post("/transcrever")
async def transcrever(imagem: UploadFile):
    conteudo = await imagem.read()
    return {
        "nome_arquivo": imagem.filename,
        "tipo": imagem.content_type,
        "tamanho_bytes": len(conteudo)
    }