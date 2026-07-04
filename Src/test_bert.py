import torch
from transformers import BertTokenizer, BertModel

class BertEmbedding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        outputs = self.model.embeddings(
            input_ids=inputs['input_ids'],
            token_type_ids=inputs['token_type_ids'],
        )
        return outputs

b = BertEmbedding()
res = b(["hello world", "this is a test sentence that is a bit longer"])
print("type:", type(res))
if isinstance(res, torch.Tensor):
    print("shape:", res.shape)
elif isinstance(res, tuple):
    print("tuple len:", len(res))
    for i, x in enumerate(res):
        if hasattr(x, 'shape'):
            print(f"element {i} shape: {x.shape}")
