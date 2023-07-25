## Fine-tuning a BART Model for Abstractive Summarization of r/relationships
### [Check out the web demo!](https://sotskopa-text-summarizer.hf.space)
### Authors
Viggo Svärdkrona

Marcus Gisslén
### Description
This repository is the result of our efforts to create an abstractive text summarizer, designed specifically to handle natural language found on Reddit. As a base, we use a [pretrained BART](https://huggingface.co/facebook/bart-base), which is then fine-tuned on `(text, summary)` pairs sourced from the [r/relationships](https://www.reddit.com/r/relationships) subreddit. This data is obtained as a subset of the [Webis-TLDR-17 Corpus](https://webis.de/data/webis-tldr-17.html).  
### Example
#### User text
*Plain and simple I’ve been broken up with my ex for
2 months now, and everyday I still think about her and
try and convince myself i’m better off without her, but I
always end up thinking about her with another guy and
it crushes me. I’ve done everything to try and get over
her. I’ve hung out with friends, changed my routine, spent
time with family, picked up hobbies I also avoid her social
media and never talk to her anymore. ...etc But I still feel
so depressed that it makes me sick and I don’t want to feel
this way anymore. I’ve also always been a lonely guy, so
when my ex left me it felt like I lost a huge part of my life.*

#### User summary
*I need help coping with my post break up depression!*

#### Generated summary
*How do I get over my ex?*

### Training locally

Run the program with `python src/main.py`.

This will download the Webis-TLDR-17 Corpus, which contains ~17 GB of data, as well as ~1 GB of parameters for the pre-trained BART. By default, the program trains the model on all datapoints in r/relationships, and saves the model in `./models/finetuned`. Optionally, you can include the following flags:

`-d <DATA>` trains the model on `<DATA>` datapoints. Note that `<DATA>` should be at most ~230,000.  
`-l <PATH>` loads a saved model from `<PATH>`.

Our fine-tuned model can be downloaded [here.](https://drive.google.com/file/d/1SSe_sIB7dCyqWvWTS1V2bKv7Sz-snNYE/view?usp=sharing)
