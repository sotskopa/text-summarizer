from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainerCallback
import datasets
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

MODEL_NAME = "facebook/bart-base"
MODEL_PATH = "models/" + MODEL_NAME
DATASET_PATH = "reddit" # 

EXAMPLES = [
    """I (18F) and my boyfriend (19F) have both been together for 6 months. I only found out this year that he only eats takeaways and just general unhealthy food.

It doesn’t bother me whether you are underweight or overweight, as long as you are healthy. The problem is that my boyfriend does not eat hardly any fruit, definitely no vegetables and just an overall unbalanced diet. I have tried to subtly motivate him to live a healthy lifestyle- a balanced diet, 5 a day, etc. He did seem motivated by my words and stated that he will start eating healthier. My boyfriend is a tall guy (6’0”) and should generally eat more than others, as he does.

Because of his diet, it affects the way that I view him. Before I became concerned about his health, I saw him as nothing other than perfect, and though I still see him as that it doesn’t feel the same. I am all for to eat whatever you want, and I do eat takeaways myself but only once a week at max. One time we had a takeaway and he ate so much he threw up and explained it happens often. This threw me aback and now I’m more concerned.

I don’t know what to do, or if I should let him know.""",
    """My partner and I have been together for years. Lived together for most of it.

I’ve felt for a long time that I am not seen by him as a separate person with my own wants, needs, and motivations that have nothing to do with him. I haven’t been able to articulate the feeling until very recently. But for most of our relationship I’ve been letting that affect me, the decisions I make, and how I spend my time, so much that I’ve realized I’m feeling completely drained by the relationship.

I feel like I’m not allowed to have feelings. I’m not allowed to be sick. I’m not allowed to be tired. Because if I ever express feeling these things he takes it personally. Like I only have headaches to avoid intimacy. Or I only feel tired to avoid spending time with him.

And according to him the only way for him to feel loved and like I’m attracted to him is if I’m having sex with him. Every. Single. Day. Over the course of the last month he’s escalated his behavior around this to just sleeping in another room anytime I even hint at not being immediately interested when he is.

But we have sex at least 4 times a week on average, usually more. And if not sex I’ll give him oral or hand stuff. Because if I don’t for even one day he’s back to being mopey and angry and feeling slighted.

But I also feel like I haven’t had a legitimate day off in over a month. I’m either at work, or working on everything at home. I’ve been going non-stop and just have nothing left to give. I’m done changing my behavior to keep him happy. I’m going to do what I need and want to do for myself.

I want him to go to therapy, because he has a lot of work he needs to do. If he truly can’t feel love or like he’s attractive or has worth if it’s not through sex, he needs help. If he feels like everything I do or feel is about him, he needs help. But he doesn’t see it that way. He sees it as me not caring about him.

What can I even do here? I can get him therapist visits, online or in person, for free from my job. So it’s not like he doesn’t have access to mental health care. And he has health insurance and a doctor. He takes an antidepressant and occasional anti-anxiety med but I feel like he needs more. He needs to do more than just take meds and keep being an ass. Please help.""",
    """Husband and I are both trainees in surgical subspecialties- I'm a neurosurgery resident and he's doing a sub-sub-specialty fellowship on the other side of the country for one year.

My schedule is extremely call heavy- I'm on call 24/7 for every weekday and every other weekend for the entire academic year, which is in and of itself very grueling. My program is very small, famously malignant, and has zero redundancy or back up. We don't do sick days.

I have protected research time next year, so we timed my pregnancy with our first child to allow me to deliver during that time. My husband been otherwise great and proactive about flying cross country almost every weekend to visit and help me out.

I caught some stupid flu and it's totally knocked me out. I've spent the last day in bed as much as I can, but starting tomorrow evening I am covering two hospitals in two different cities, one of which is a level 1 trauma center, as the sole neurosurgery resident, until tuesday morning. We typically get many overnight consults and operate frequently during the weekend. My coresident who I am cross covering is on vacation and heading out of town, and the fellow is also out of town now. There is no one who can help cover me.

We werent originally planning on him coming out this weekend because I was on call, but when I got sick I booked him a last minute (pretty expensive) flight and he was down to come and help take care of me, drive me around, feed me, etc. He was on his way to the airport when he found out his fellowhsip program is doing this really rare, once in a lifetime surgery this saturday. He immediately cancelled his flight.

The surgery is a huge deal in his subspecialty, but I'm so disappointed and so stressed and it's hard not to spiral and panic thinking about what the fuck we are doing bringing a baby into a two surgeon household where our lives both suck. I want to be taken care of! I want to feel like a priority! we both have jobs that to some extent, always have to come first, but I don't know how not be sad or resentful about it or scared that things will never get better.""",
]


def load_model(path=MODEL_NAME):
    """Load pretrained model and tokenizer. Uses cached model if available."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tokenizer, model


def load_dataset(path):
    """Loads dataset from specified path."""
    dataset = datasets.load_dataset(path, split="train")
    # Select only examples from the relationships subreddit
    dataset = dataset.filter(
        lambda example: example["subreddit"] == "relationships", num_proc=4
    )
    dataset = dataset.filter(
        lambda example: example["summary"] is not None
        and example["content"] is not None,
        num_proc=4,
    )

    def filter_length(example):
        summary_length = len(example["summary"].split())
        content_length = len(example["content"].split())
        return (
            summary_length > 2
            and summary_length < 128
            and content_length > 32
            and content_length < 512
        )

    dataset = dataset.filter(lambda example: filter_length(example), num_proc=4)

    print("Dataset size:", len(dataset))

    return dataset.select_columns(["content", "summary"])


def preprocess_dataset(dataset, tokenizer):
    """Preprocesses dataset for model."""

    # Define the maximum input and target length
    max_input_length = 512
    max_target_length = 128

    def tokenize(examples):
        text_column, summary_column = "content", "summary"
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
            else:
                print("Empty example found. Skipping.")

        model_inputs = tokenizer(
            inputs, max_length=max_input_length, padding="max_length", truncation=True
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(tokenize, batched=True, batch_size=128, num_proc=4)
    return dataset


class SummariesCallback(TrainerCallback):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print("Generating summaries...")
        self.generate_summaries()

    def generate_summaries(self):
        """Generates summaries from texts."""
        inputs = self.tokenizer(
            EXAMPLES,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        outputs = self.model.generate(
            input_ids, attention_mask=attention_mask, max_length=128
        )
        for i, output in enumerate(outputs):
            print(
                f"Example {i}: {self.tokenizer.decode(output, skip_special_tokens=True)}"
            )


def generate_summary(model, tokenizer, text):
    """Generates summary from text."""
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


        
    







        
            
                










