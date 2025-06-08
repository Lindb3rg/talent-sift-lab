import spacy
import random

# Load Blank Model
nlp = spacy.blank('en')

def train_model(train_data):
    # Add NER pipeline if it doesn't exist
    if 'ner' not in nlp.pipe_names:
        # Use the string name instead of create_pipe
        nlp.add_pipe('ner', last=True)
    
    # Get the NER component
    ner = nlp.get_pipe('ner')
    
    # Add labels in the NLP pipeline
    for _, annotation in train_data:
        for ent in annotation.get('entities'):
            ner.add_label(ent[2])
    
    # Remove other pipelines if they are there
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # Initialize the model
        nlp.initialize()
        
        for itn in range(10):  # train for 10 iterations
            print("Starting iteration " + str(itn))
            random.shuffle(train_data)
            losses = {}
            
            for text, annotations in train_data:
                try:
                    # Create Example objects for training
                    doc = nlp.make_doc(text)
                    example = spacy.training.Example.from_dict(doc, annotations)
                    
                    nlp.update(
                        [example],  # batch of Example objects
                        drop=0.2,  # dropout - make it harder to memorise data
                        losses=losses
                    )
                except Exception as e:
                    print(f"Error processing: {text[:50]}... - {e}")
                    
            print(losses)

# # Start Training model
# train_model(train_data)