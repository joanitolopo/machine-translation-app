from numpy.lib.utils import source
import torch
from translate import Translator

import numpy as np
import random

from arsitektur.encoder import Encoder
from arsitektur.decoder import Decoder
from arsitektur.transformer import Transformer
from utils import translate_sentence, tokenize_src, tokenize_trg

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prediksi(source, target, sentence, max_len=100):

    # special case for english
    if source == "English":
        translator= Translator(from_lang="english",to_lang="indonesian")
        sentence = translator.translate(sentence)
        source="Indonesia"
    print(target)
    
    #load weights
    weights = torch.load(f"model/{str(target)}/weights.pth", map_location='cpu')
    config = torch.load(f"model/{str(target)}/configs.pth", map_location='cpu')
    vocab_src = torch.load(f"data/{str(target)}/source_vocab.pth", map_location='cpu')        
    vocab_trg = torch.load(f"data/{str(target)}/target_vocab.pth", map_location='cpu')
    
    
    # Load arsitektur
    enc = Encoder(config.INPUT_DIM, config.HID_DIM, config.ENC_LAYERS, config.ENC_HEADS, 
                config.ENC_PF_DIM, config.ENC_DROPOUT, device)
    dec = Decoder(config.OUTPUT_DIM, config.HID_DIM, config.DEC_LAYERS, config.DEC_HEADS, 
                config.DEC_PF_DIM, config.DEC_DROPOUT, device)

    SRC_PAD_IDX = vocab_src.vocab.stoi[vocab_src.pad_token]
    TRG_PAD_IDX = vocab_trg.vocab.stoi[vocab_trg.pad_token]
    
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    # load model

    model.load_state_dict(weights)
    model = model.to(device)

    #predict
    translation, _ = translate_sentence([sentence], vocab_src, vocab_trg, model, device, max_len=max_len)
    del translation[-1]
    result = " ".join(translation)

    return result




