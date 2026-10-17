# ============================================================
#  Sketchly – load‑what‑you‑saved edition (fixed indentation)
# ============================================================

import numpy as np
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Install with  ➜  pip install streamlit-drawable-canvas")
    st.stop()

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import openai

# Paths & keys 
MODEL_FILE  = r"C:\Users\vabss\Downloads\efficientnet_final_model.h5"
NUM_CLASSES = 345
openai.api_key = "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"

#
# 

@st.cache_resource(show_spinner="Loading model …")
def load_trained_model():
    # A) Try loading the entire model exactly as saved
    try:
        return tf.keras.models.load_model(MODEL_FILE, compile=False, safe_mode=False)
    except Exception as e_full:
        st.info(f"Full load failed – rebuilding backbone: {e_full}")

    # B) Rebuild the architecture and load matching weights
    base = EfficientNetV2B0(include_top=False, input_shape=(224, 224, 3), weights=None)
    base.trainable = False
    model = models.Sequential([
        layers.Lambda(lambda x: x),           # placeholder for any augmentation
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    # Create variables, then load whatever weights match
    model(tf.zeros([1, 224, 224, 3]))
    model.load_weights(MODEL_FILE, by_name=True, skip_mismatch=True)
    return model

model = load_trained_model()

# 
# 2) Class names 

class_names = [     "aircraft carrier",
    "airplane",
    "alarm clock",
    "ambulance",
    "angel",
    "animal migration",
    "ant",
    "anvil",
    "apple",
    "arm",
    "asparagus",
    "axe",
    "backpack",
    "banana",
    "bandage",
    "barn",
    "baseball",
    "baseball bat",
    "basket",
    "basketball",
    "bat",
    "bathtub",
    "beach",
    "bear",
    "beard",
    "bed",
    "bee",
    "belt",
    "bench",
    "bicycle",
    "binoculars",
    "bird",
    "birthday cake",
    "blackberry",
    "blueberry",
    "book",
    "boomerang",
    "bottlecap",
    "bowtie",
    "bracelet",
    "brain",
    "bread",
    "bridge",
    "broccoli",
    "broom",
    "bucket",
    "bulldozer",
    "bus",
    "bush",
    "butterfly",
    "cactus",
    "cake",
    "calculator",
    "calendar",
    "camel",
    "camera",
    "camouflage",
    "campfire",
    "candle",
    "cannon",
    "canoe",
    "car",
    "carrot",
    "castle",
    "cat",
    "ceiling fan",
    "cello",
    "cell phone",
    "chair",
    "chandelier",
    "church",
    "circle",
    "clarinet",
    "clock",
    "cloud",
    "coffee cup",
    "compass",
    "computer",
    "cookie",
    "cooler",
    "couch",
    "cow",
    "crab",
    "crayon",
    "crocodile",
    "crown",
    "cruise ship",
    "cup",
    "diamond",
    "dishwasher",
    "diving board",
    "dog",
    "dolphin",
    "donut",
    "door",
    "dragon",
    "dresser",
    "drill",
    "drums",
    "duck",
    "dumbbell",
    "ear",
    "elbow",
    "elephant",
    "envelope",
    "eraser",
    "eye",
    "eyeglasses",
    "face",
    "fan",
    "feather",
    "fence",
    "finger",
    "fire hydrant",
    "fireplace",
    "firetruck",
    "fish",
    "flamingo",
    "flashlight",
    "flip flops",
    "floor lamp",
    "flower",
    "flying saucer",
    "foot",
    "fork",
    "frog",
    "frying pan",
    "garden",
    "garden hose",
    "giraffe",
    "goatee",
    "golf club",
    "grapes",
    "grass",
    "guitar",
    "hamburger",
    "hammer",
    "hand",
    "harp",
    "hat",
    "headphones",
    "hedgehog",
    "helicopter",
    "helmet",
    "hexagon",
    "hockey puck",
    "hockey stick",
    "horse",
    "hospital",
    "hot air balloon",
    "hot dog",
    "hot tub",
    "hourglass",
    "house",
    "house plant",
    "hurricane",
    "ice cream",
    "jacket",
    "jail",
    "kangaroo",
    "key",
    "keyboard",
    "knee",
    "knife",
    "ladder",
    "lantern",
    "laptop",
    "leaf",
    "leg",
    "light bulb",
    "lighter",
    "lighthouse",
    "lightning",
    "line",
    "lion",
    "lipstick",
    "lobster",
    "lollipop",
    "mailbox",
    "map",
    "marker",
    "matches",
    "megaphone",
    "mermaid",
    "microphone",
    "microwave",
    "monkey",
    "moon",
    "mosquito",
    "motorbike",
    "mountain",
    "mouse",
    "moustache",
    "mouth",
    "mug",
    "mushroom",
    "nail",
    "necklace",
    "nose",
    "ocean",
    "octagon",
    "octopus",
    "onion",
    "oven",
    "owl",
    "paintbrush",
    "paint can",
    "palm tree",
    "panda",
    "pants",
    "paper clip",
    "parachute",
    "parrot",
    "passport",
    "peanut",
    "pear",
    "peas",
    "pencil",
    "penguin",
    "piano",
    "pickup truck",
    "picture frame",
    "pig",
    "pillow",
    "pineapple",
    "pizza",
    "pliers",
    "police car",
    "pond",
    "pool",
    "popsicle",
    "postcard",
    "potato",
    "power outlet",
    "purse",
    "rabbit",
    "raccoon",
    "radio",
    "rain",
    "rainbow",
    "rake",
    "remote control",
    "rhinoceros",
    "rifle",
    "river",
    "roller coaster",
    "rollerskates",
    "sailboat",
    "sandwich",
    "saw",
    "saxophone",
    "school bus",
    "scissors",
    "scorpion",
    "screwdriver",
    "sea turtle",
    "see saw",
    "shark",
    "sheep",
    "shoe",
    "shorts",
    "shovel",
    "sink",
    "skateboard",
    "skull",
    "skyscraper",
    "sleeping bag",
    "smiley face",
    "snail",
    "snake",
    "snorkel",
    "snowflake",
    "snowman",
    "soccer ball",
    "sock",
    "speedboat",
    "spider",
    "spoon",
    "spreadsheet",
    "square",
    "squiggle",
    "squirrel",
    "stairs",
    "star",
    "steak",
    "stereo",
    "stethoscope",
    "stitches",
    "stop sign",
    "stove",
    "strawberry",
    "streetlight",
    "string bean",
    "submarine",
    "suitcase",
    "sun",
    "swan",
    "sweater",
    "swing set",
    "sword",
    "syringe",
    "table",
    "teapot",
    "teddy-bear",
    "telephone",
    "television",
    "tennis racquet",
    "tent",
    "The Eiffel Tower",
    "The Great Wall of China",
    "The Mona Lisa",
    "tiger",
    "toaster",
    "toe",
    "toilet",
    "tooth",
    "toothbrush",
    "toothpaste",
    "tornado",
    "tractor",
    "traffic light",
    "train",
    "tree",
    "triangle",
    "trombone",
    "truck",
    "trumpet",
    "t-shirt",
    "umbrella",
    "underwear",
    "van",
    "vase",
    "violin",
    "washing machine",
    "watermelon",
    "waterslide",
    "whale",
    "wheel",
    "windmill",
    "wine bottle",
    "wine glass",
    "wristwatch",
    "yoga",
    "zebra",
    "zigzag"]

# 3) Streamlit UI

st.set_page_config(page_title="Sketchly", page_icon="✏️")
st.title("🖌️ Sketchly – draw & recognize")

with st.sidebar:
    st.markdown("1. Draw on canvas  \n2. Click **Predict**  \n3. See guess + GPT tips")

canvas = st_canvas(
    fill_color="rgba(0,0,0,0)", stroke_width=8, stroke_color="#000",
    background_color="#FFF", height=400, width=400,
    drawing_mode="freedraw", key="canvas"
)

left, right = st.columns(2)

with left:
    if st.button("Predict", type="primary"):
        if canvas.image_data is None:
            st.warning("Please draw something first!")
        else:
            img = Image.fromarray(canvas.image_data.astype(np.uint8), mode="RGBA")\
                     .convert("RGB").resize((224, 224))
            arr = preprocess_input(np.array(img, dtype=np.float32))[None, ...]
            probs = model(arr, training=False).numpy()[0]
            idx   = int(probs.argmax())
            label = class_names[idx] if idx < len(class_names) else f"Class {idx}"

            st.subheader("Prediction")
            st.success(label.capitalize())

            # GPT suggestions
            try:
                tips = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user",
                               "content": f"Two short bullet tips to improve a doodle of a '{label}'."}],
                    max_tokens=60, temperature=0.7
                ).choices[0].message.content.strip()
            except Exception:
                tips = "• Use darker outlines.\n• Emphasize key features."

            st.subheader("Suggestions")
            st.markdown(tips)

with right:
    if st.button("Clear"):
        st.experimental_rerun()
