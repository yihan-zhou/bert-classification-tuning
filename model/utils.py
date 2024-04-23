from enum import Enum

from model.sequence_classification import (
    BertPrefixForSequenceClassification,
)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


class TaskType(Enum):
    TOKEN_CLASSIFICATION = (1,)
    SEQUENCE_CLASSIFICATION = (2,)
    QUESTION_ANSWERING = (3,)
    MULTIPLE_CHOICE = 4


PREFIX_MODELS = {
    "bert": {TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification}
}

AUTO_MODELS = {TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification}


def get_model(model_args, task_type: TaskType, config: AutoConfig):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        model_class = PREFIX_MODELS[config.model_type][task_type]

        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0

        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print("***** total param is {} *****".format(total_param))

    return model
