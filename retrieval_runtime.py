"""Faithful option-wise hybrid retrieval from the FailureSensorIQ notebook.

Uses only the 2,133-question training split for retrieval memory.
TF-IDF + all-MiniLM-L6-v2, separately min-max normalized per query,
alpha=0.10, k=2, contradiction weight=0.75.
"""
import json
import re
import threading
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
_INDEX = None
_LOCK = threading.Lock()


def clean_text(x):
    return re.sub(r'\s+', ' ', str(x or '')).strip()


def make_option_text(row, option):
    return ' | '.join([
        f"asset: {clean_text(row.get('asset_name', 'unknown'))}",
        f"task: {clean_text(row.get('relevancy', 'unknown'))}",
        f"question type: {clean_text(row.get('question_type', 'unknown'))}",
        f"question: {clean_text(row.get('question', ''))}",
        f"candidate option: {clean_text(option)}",
    ])


def minmax01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-12 else (x-lo)/(hi-lo)


def topk_mean(x, k=2):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return 0.0
    n = min(k,len(x))
    return float(x[np.argpartition(x,-n)[-n:]].mean())


class HybridRetrieval:
    def __init__(self, rows, semantic_model=None):
        texts=[]; labels=[]
        for row in rows:
            options=row['options']; correct=row['correct']
            if len(options)!=len(correct): continue
            for option, gold in zip(options,correct):
                texts.append(make_option_text(row,option))
                labels.append(int(bool(gold)))
        self.labels=np.asarray(labels,dtype=np.int8)
        self.lex=TfidfVectorizer(lowercase=True,stop_words=None,ngram_range=(1,2),
                                 min_df=1,sublinear_tf=True,norm='l2')
        self.lex_matrix=self.lex.fit_transform(texts)
        if semantic_model is None:
            from sentence_transformers import SentenceTransformer
            # Explicitly stay on CPU; ZeroGPU allocations are only for the QLoRA handler.
            semantic_model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
        self.sem=semantic_model
        self.emb=self.sem.encode(texts,batch_size=64,show_progress_bar=False,
                                 convert_to_numpy=True,normalize_embeddings=True).astype(np.float32)

    def predict(self,row):
        opts=list(row['options']); queries=[make_option_text(row,opt) for opt in opts]
        qlex=self.lex.transform(queries) @ self.lex_matrix.T
        qemb=self.sem.encode(queries,convert_to_numpy=True,
                              normalize_embeddings=True,show_progress_bar=False).astype(np.float32)
        sem=qemb @ self.emb.T
        scores=[]; details=[]
        for i,option in enumerate(opts):
            lex=np.asarray(qlex.getrow(i).toarray()).ravel()
            similarity=0.10*minmax01(lex)+0.90*minmax01(sem[i])
            support=topk_mean(similarity[self.labels==1],2)
            contradiction=topk_mean(similarity[self.labels==0],2)
            score=support-0.75*contradiction
            scores.append(score)
            details.append({'option':'ABCDE'[i], 'score':round(score,5),
                            'support':round(support,5),'contradiction':round(contradiction,5)})
        idx=int(np.argmax(scores)); s=np.sort(scores)
        return {'label':'ABCDE'[idx], 'answer':opts[idx], 'details':details,
                'margin':float(s[-1]-s[-2]) if len(s)>1 else 0.0}


def get_index():
    global _INDEX
    if _INDEX is None:
        with _LOCK:
            if _INDEX is None:
                with (ROOT/'data/retrieval_train.jsonl').open(encoding='utf-8') as f:
                    rows=[json.loads(line) for line in f if line.strip()]
                _INDEX=HybridRetrieval(rows)
    return _INDEX
