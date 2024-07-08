import argparse
from enum import Enum
from dotenv import load_dotenv
import os
import logging
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from commands.chat import chat
from commands.create_db import create_db
from commands.import_data import import_data

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Command(Enum):
    CREATE_DB = "create-db"
    IMPORT_DATA = "import-data"
    CHAT = "chat"


def main():
    parser = argparse.ArgumentParser(description="Application Description")

    subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="command",
        help="Display available subcommands",
    )

    # create-db command
    subparsers.add_parser(
        Command.CREATE_DB.value, help="Create a database"
    ).set_defaults(func=create_db)

    # import-data command
    import_data_parser = subparsers.add_parser(
        Command.IMPORT_DATA.value, help="Import data"
    )
    import_data_parser.add_argument(
        "data_source", type=str, help="Specify the PDF data source"
    )
    import_data_parser.set_defaults(func=import_data)

    # chat command
    chat_parser = subparsers.add_parser(
        Command.CHAT.value, help="Use chat feature"
    )
    chat_parser.set_defaults(func=chat)

    args = parser.parse_args()
    
    logger.info(f"Running command: {args.command}")

    if hasattr(args, "func"):
        if torch.cuda.is_available():
            device = "cuda"
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info("CUDA is available, using GPU")
        else:
            device = "cpu"
            bnb_config = None
            logger.info("CUDA is not available, using CPU")
                
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                os.getenv("TOKENIZER_NAME"),
                token=os.getenv("HUGGING_FACE_ACCESS_TOKEN"),
            )
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error("Failed to load tokenizer", exc_info=e)
            return
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                os.getenv("MODEL_NAME"),
                token=os.getenv("HUGGING_FACE_ACCESS_TOKEN"),
                quantization_config=bnb_config,
                device_map=device,
                torch_dtype=torch.float16,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model", exc_info=e)
            return

        try:
            args.func(args, model, device, tokenizer)
        except Exception as e:
            logger.error(f"Failed to execute command {args.command}", exc_info=e)
    else:
        logger.error("Invalid command. Use '--help' for assistance.")


if __name__ == "__main__":
    main()
