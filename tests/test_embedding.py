import torch 

from models.embeddings import TokenEmbedding

def test_embedding_output_shape():
    embedding = TokenEmbedding(
        vocab_size=12000,
        hidden_dim=512,
        pad_token_id=0, 
    )

    input_ids = torch.randint(0, 12000, (2, 128))
    output = embedding(input_ids)

    assert output.shape == (2, 128, 512)

def test_embedding_is_trainable():
    embedding = TokenEmbedding(
        vocab_size=100, 
        hidden_dim=32, 
        pad_token_id=0,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    output = embedding(input_ids)

    loss = output.sum()
    loss.backward()

    assert embedding.embedding.weight.grad is not None

def test_padding_embedding_has_no_gradient():
    embedding = TokenEmbedding(
        vocab_size=10,
        hidden_dim=4,
        pad_token_id=0,
    )

    input_ids = torch.tensor([[0, 1, 2]])
    output = embedding(input_ids)

    output.sum().backward()

    pad_grad = embedding.embedding.weight.grad[0]

    assert torch.all(pad_grad == 0)
