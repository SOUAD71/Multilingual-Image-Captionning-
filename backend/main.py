from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn
from pydantic import BaseModel
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Captioning API", version="1.0.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des modèles
class ModelLoader:
    def __init__(self):
        self.blip_processor = None
        self.blip_model = None
        self.nllb_tokenizer = None
        self.nllb_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Utilisation du device: {self.device}")
        
    def load_blip(self):
        if self.blip_processor is None:
            logger.info("Chargement du modèle BLIP...")
            self.blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            logger.info("Modèle BLIP chargé avec succès")
    
    def load_nllb(self):
        if self.nllb_tokenizer is None:
            logger.info("Chargement du modèle NLLB...")
            try:
                # Essayer d'abord avec use_fast=False pour avoir lang_code_to_id
                self.nllb_tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/nllb-200-distilled-600M",
                    use_fast=False,
                    src_lang="eng_Latn"
                )
                logger.info("Tokenizer NLLB (slow) chargé")
            except Exception as e:
                logger.warning(f"Échec chargement tokenizer slow: {e}")
                # Fallback sur tokenizer rapide
                self.nllb_tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/nllb-200-distilled-600M",
                    src_lang="eng_Latn"
                )
                logger.info("Tokenizer NLLB (fast) chargé")
            
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/nllb-200-distilled-600M"
            ).to(self.device)
            logger.info("Modèle NLLB chargé avec succès")

model_loader = ModelLoader()

# Mapping des langues pour NLLB
LANGUAGE_CODES = {
    "français": "fra_Latn",
    "anglais": "eng_Latn",
    "espagnol": "spa_Latn",
    "allemand": "deu_Latn",
    "italien": "ita_Latn",
    "portugais": "por_Latn",
    "arabe": "arb_Arab",
    "chinois": "zho_Hans",
    "japonais": "jpn_Jpan",
    "russe": "rus_Cyrl"
}

class TranslationRequest(BaseModel):
    text: str
    target_language: str

@app.on_event("startup")
async def startup_event():
    """Préchargement des modèles au démarrage"""
    model_loader.load_blip()
    model_loader.load_nllb()

@app.get("/")
async def root():
    return {
        "message": "Image Captioning API",
        "endpoints": {
            "caption": "/api/caption",
            "translate": "/api/translate",
            "languages": "/api/languages"
        }
    }

@app.get("/api/languages")
async def get_languages():
    """Retourne la liste des langues supportées"""
    return {
        "languages": list(LANGUAGE_CODES.keys()),
        "total": len(LANGUAGE_CODES)
    }

@app.post("/api/caption")
async def generate_caption(file: UploadFile = File(...)):
    """
    Génère une description textuelle d'une image uploadée
    """
    try:
        # Vérification du type de fichier
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="Le fichier doit être une image"
            )
        
        # Lecture et traitement de l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Génération du caption
        model_loader.load_blip()
        inputs = model_loader.blip_processor(image, return_tensors="pt").to(
            model_loader.device
        )
        
        with torch.no_grad():
            out = model_loader.blip_model.generate(**inputs, max_length=50)
        
        caption = model_loader.blip_processor.decode(out[0], skip_special_tokens=True)
        
        logger.info(f"Caption généré: {caption}")
        
        return JSONResponse({
            "success": True,
            "caption": caption,
            "language": "anglais"
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du caption: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate")
async def translate_text(request: TranslationRequest):
    """
    Traduit un texte vers une langue cible
    """
    try:
        if request.target_language not in LANGUAGE_CODES:
            raise HTTPException(
                status_code=400,
                detail=f"Langue non supportée. Langues disponibles: {list(LANGUAGE_CODES.keys())}"
            )
        
        model_loader.load_nllb()
        
        # Configuration de la traduction
        source_lang = "eng_Latn"  # Le caption BLIP est en anglais
        target_lang = LANGUAGE_CODES[request.target_language]
        
        # Préparation du tokenizer avec la langue source
        model_loader.nllb_tokenizer.src_lang = source_lang
        
        # Encodage du texte
        inputs = model_loader.nllb_tokenizer(
            request.text, 
            return_tensors="pt",
            padding=True
        ).to(model_loader.device)
        
        # Génération de la traduction avec forced_bos_token_id
        with torch.no_grad():
            # Méthode robuste pour obtenir l'ID du token de langue cible
            # On utilise le vocabulaire du tokenizer directement
            try:
                # Essayer d'abord avec lang_code_to_id (tokenizer lent)
                if hasattr(model_loader.nllb_tokenizer, 'lang_code_to_id'):
                    forced_bos_token_id = model_loader.nllb_tokenizer.lang_code_to_id[target_lang]
                else:
                    # Sinon utiliser convert_tokens_to_ids
                    forced_bos_token_id = model_loader.nllb_tokenizer.convert_tokens_to_ids(target_lang)
            except (KeyError, AttributeError) as e:
                # Solution de secours : encoder le code de langue
                logger.warning(f"Utilisation de la méthode de secours pour {target_lang}")
                forced_bos_token_id = model_loader.nllb_tokenizer.encode(target_lang, add_special_tokens=False)[0]
            
            logger.info(f"Token ID pour {target_lang}: {forced_bos_token_id}")
            
            translated_tokens = model_loader.nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=100,
                num_beams=5,
                early_stopping=True
            )
        
        # Décodage de la traduction
        translated_text = model_loader.nllb_tokenizer.batch_decode(
            translated_tokens, 
            skip_special_tokens=True
        )[0]
        
        logger.info(f"Traduction: {request.text} -> {translated_text}")
        
        return JSONResponse({
            "success": True,
            "original_text": request.text,
            "translated_text": translated_text,
            "source_language": "anglais",
            "target_language": request.target_language
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la traduction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/caption-and-translate")
async def caption_and_translate(
    file: UploadFile = File(...),
    target_language: str = "français"
):
    """
    Génère un caption et le traduit en une seule requête
    """
    try:
        # Génération du caption
        caption_response = await generate_caption(file)
        caption_data = caption_response.body.decode()
        
        import json
        caption_json = json.loads(caption_data)
        caption = caption_json["caption"]
        
        # Traduction
        if target_language != "anglais":
            translation_request = TranslationRequest(
                text=caption,
                target_language=target_language
            )
            translation_response = await translate_text(translation_request)
            translation_data = translation_response.body.decode()
            translation_json = json.loads(translation_data)
            
            return JSONResponse({
                "success": True,
                "original_caption": caption,
                "translated_caption": translation_json["translated_text"],
                "source_language": "anglais",
                "target_language": target_language
            })
        else:
            return JSONResponse({
                "success": True,
                "original_caption": caption,
                "translated_caption": caption,
                "source_language": "anglais",
                "target_language": "anglais"
            })
            
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)