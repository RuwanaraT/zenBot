import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

class MovieDialogDataset(Dataset):
    def __init__(self, lines_file, conversations_file):
        self.load_dataset(lines_file, conversations_file)

    def load_dataset(self, lines_file, conversations_file):
        lines = {}
        with open(lines_file, "r", encoding="iso-8859-1") as file:
            for line in file:
                parts = line.strip().split(" +++$+++ ")
                if len(parts) == 5:
                    line_id = int(parts[0][1:])
                    dialog_id = parts[1]
                    character_id = parts[2]
                    text = parts[4]
                    lines[line_id] = {
                        "dialog_id": dialog_id,
                        "character_id": character_id,
                        "text": text
                    }

        dialogues = []
        with open(conversations_file, "r", encoding="iso-8859-1") as file:
            for line in file:
                parts = line.strip().split(" +++$+++ ")
                if len(parts) > 4:
                    dialogue_ids = eval(parts[3])
                    for i in range(len(dialogue_ids) - 1):
                        input_line_id = dialogue_ids[i]
                        target_line_id = dialogue_ids[i + 1]
                        input_text = lines[input_line_id]["text"].strip()
                        target_text = lines[target_line_id]["text"].strip()
                        if input_text and target_text:
                            dialogues.append({
                                "input_text": input_text,
                                "target_text": target_text
                            })

        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, index):
        return self.dialogues[index]

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = MovieDialogDataset(
    lines_file="C:/Users/dell/Desktop/AI_Virtual_C/cornels_dataset/movie_lines.tsv",
    conversations_file="C:/Users/dell/Desktop/AI_Virtual_C/cornels_dataset/movie_conversations.tsv"
)

# Create data loader
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Set the model in training mode
model.train()

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        # Preprocess the input and target sequences
        inputs = tokenizer.batch_encode_plus(
            batch["input_text"],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        targets = tokenizer.batch_encode_plus(
            batch["target_text"],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=targets["input_ids"],
            decoder_attention_mask=targets["attention_mask"]
        )

        # Compute the loss
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for the epoch
    print(f"Epoch {epoch+1} Loss: {total_loss / len(data_loader)}")

# Save the trained model
model.save_pretrained("dialogue_model")
tokenizer.save_pretrained("tokenizer")
