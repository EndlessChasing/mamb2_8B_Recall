# Hugging Face publication

The verified custom Resurface adapter and its runtime are public at
[EndlessChasing/Mamb2_8B_Recall](https://huggingface.co/EndlessChasing/Mamb2_8B_Recall).
This package contains 34 files totaling 5,859,085 bytes, including the exact
2,374,143-byte adapter, source code, model card and original quality receipts.
The NVIDIA base weights and tokenizer are downloaded separately as described
in the model card.

The public commit is `5b7b27af9cd49d2a55bd080383f1d873c181ce83`. Download that
verified version with:

```bash
hf download EndlessChasing/Mamb2_8B_Recall \
  --revision 5b7b27af9cd49d2a55bd080383f1d873c181ce83 \
  --local-dir mamb2-recall
```

The code and quality receipts in the Hub package come from GitHub commit
`4478c034aff9e5dab7af47efa84214287824138d`. The Hub-specific card and
`resurface_config.json` identify the separate pinned base model and custom
adapter loader. This is an adapter distribution; the measured model still
requires the full base weights during execution.

All 34 remote file sizes and Git/LFS identities matched the prepared package.
Every file was then downloaded through an anonymous HTTPS request at the
pinned commit and checked by SHA-256. The adapter remains:

`e8b2b4dfe69f8e85dc9e147c9aeaa3297cff14f4043e558bb1795ad476c1fca0`

The checksum manifest SHA-256 is:

`3faebc20566aadb53bf3f8871ecf9a8b426b84999a771ccf35d0e84fb304c541`

See the [preparation record](../reports/huggingface_preparation_v1.json) and
[public download verification](../reports/huggingface_publication_verification_v1.json).
These checks establish distribution integrity. Model-quality evidence is in
[VERIFICATION.md](VERIFICATION.md): PPL 7.0520635156 and normal MK 365/384
under the documented protocol.
