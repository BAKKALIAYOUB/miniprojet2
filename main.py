import base64
import jwt
from datetime import datetime, timedelta
from fastapi import FastAPI, Body, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pandas as pd
import numpy as np
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from CalculeCaracteristics import caracGlobale
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter
import os
from io import BytesIO
from pymongo import MongoClient
from pydantic import BaseModel
import bcrypt
from fastapi import Depends, HTTPException, status
from typing import Optional

# Initialize the FastAPI app
app = FastAPI(debug=True)
app.mount("/images", StaticFiles(directory="RSSCN7"), name="images")
app.mount("/uploadSearch", StaticFiles(directory="uploadSearch"), name="uploadSearch")

# Directory to save uploaded files
UPLOAD_DIR = "upload"
UPLOAD_DIR2 = "uploadSearch"
WEIGHTS_PATH = "./weights.npy"
FEATURES_PATH = './image_features.csv'
ALPHA = 0.8
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR2, exist_ok=True)

if not os.path.exists(WEIGHTS_PATH):
    features = pd.read_csv(FEATURES_PATH).iloc[:, 2:]
    initial_weights = np.ones(features.shape)
    np.save(WEIGHTS_PATH, initial_weights)

# MongoDB client
client = MongoClient("mongodb://localhost:27017")
db = client['users']
users_collection = db['users']


# Pydantic models for user registration
class UserRegister(BaseModel):
    username: str
    password: str
    email: str
    phone: str


# Allow CORS for the frontend (localhost:3000)
origins = [
    "http://localhost:3000",  # Allow your frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Autorisez votre frontend
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes
    allow_headers=["*"],  # Autorise tous les headers
)


@app.get("/download/{category}/{image_path}")
def download_image(category: str, image_path: str):
    # Construct the full file path including category
    file_path = os.path.join("RSSCN7", category, image_path)

    # Check if the file exists
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    # Serve the file
    return FileResponse(file_path, media_type="application/octet-stream", filename=image_path)


# User registration route
@app.post("/registeration")
async def register(user: UserRegister):
    # Check if the user already exists
    existing_user = users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    # Hash the password
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    # Save user to the database
    users_collection.insert_one({
        "username": user.username,
        "password": hashed_password,
        "email": user.email,
        "phone": user.phone
    })
    return JSONResponse(content={"message": "User registered successfully"}, status_code=200)


# Function to create JWT token
def create_jwt(user_id: str) -> str:
    expiration = datetime.utcnow() + timedelta(hours=1)  # Token expiration time
    payload = {"user_id": user_id, "exp": expiration}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Find the user by username
    user = users_collection.find_one({"username": form_data.username})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    # Check if the password matches
    if not bcrypt.checkpw(form_data.password.encode('utf-8'), user['password']):
        raise HTTPException(status_code=400, detail="Invalid username or password")
    # Generate JWT token
    payload = {"user_id": str(user["_id"])}  # Adjust based on your user model
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return JSONResponse(content={"message": "Login successful", "token": token}, status_code=200)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# Secret key used for encoding and decoding the JWT
SECRET_KEY = "your_secret_key"  # Replace with your actual secret key
ALGORITHM = "HS256"  # Replace with your algorithm if different


# Function to decode the JWT and extract the user_id
def decode_jwt(token: str) -> Optional[str]:
    try:
        # Decode the token with the secret key and algorithm
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Extract the user_id (or any field you have in your JWT payload)
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# Dependency to extract the current user_id from the token
async def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    user_id = decode_jwt(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user_id


# Function to calculate distance between feature vectors (Euclidean or Cosine)
def calcule_distance(vect1, vect2, weights=None):
    """
    Calculate the distance between two vectors with optional weighted distance.

    Args:
    vect1 (np.array): First feature vector
    vect2 (np.array): Second feature vector
    weights (np.array, optional): Weight array for features

    Returns:
    float: Weighted Euclidean distance
    """
    if weights is not None:
        # Apply weights element-wise to each vector
        weighted_vect1 = vect1 * weights[0]  # Use the first row of weights
        weighted_vect2 = vect2 * weights[0]  # Use the first row of weights

        # Calculate distance using weighted vectors
        dist = np.linalg.norm(weighted_vect1 - weighted_vect2)
    else:
        # If no weights, calculate standard Euclidean distance
        dist = np.linalg.norm(vect1 - vect2)

    return dist

@app.delete("/deleteImage")
async def delete_image(image_url: str, authorization: str = Header(None)):
    """
    Endpoint to delete a specific image uploaded by the user.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Extract image path from the URL
    path_parts = image_url.split("/")
    category = path_parts[-2]
    filename = path_parts[-1]

    user_dir = UPLOAD_DIR3 / Path(user_id) / category
    image_path = user_dir / filename

    # Check if the image file exists
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        # Delete the image
        os.remove(image_path)
        return JSONResponse(content={"message": "Image deleted successfully"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error deleting image")
def getSimilarImages(imagePath):
    """
    Get similar images from the database by comparing feature vectors.
    """
    # Read the uploaded image
    img = cv2.imread(imagePath)

    # Extract features for the uploaded image
    carac = caracGlobale(img).flatten()

    # Load the features of all images from the CSV file
    images_features = pd.read_csv(FEATURES_PATH)
    features = images_features.iloc[:, 2:].values

    # Load weights
    weights = np.load(WEIGHTS_PATH)

    # Calculate the weighted distances
    # Calculate the weighted distances
    distances = np.array([
        calcule_distance(carac, feat, weights=weights[index:index + 1])  # Pass weights for this specific row
        for index, feat in enumerate(features)
    ])

    # Add the distances to the dataframe
    images_features['distance'] = distances

    # Sort images by distance (smallest distance = most similar)
    similar_images = images_features.sort_values(by='distance').head(10)

    # Ensure the full image path is returned
    base_url = "http://127.0.0.1:8000/images"
    similar_images['ImagePath'] = similar_images['ImagePath'].apply(
        lambda x: x.replace("\\", "/").lstrip("/")  # Remove leading slash and replace backslashes
    )

    similar_images = similar_images.reset_index()
    return similar_images[['index', 'ImagePath', 'Category', 'distance']]

@app.get("/getImages_for_search")
async def get_images(authorization: str = Header(None)):
    """
    Endpoint to get the images uploaded by the current user, including images outside categories.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Check if user directory exists
    user_dir = Path(UPLOAD_DIR3) / Path(user_id)

    if not user_dir.exists():
        # If directory doesn't exist, return no images
        return JSONResponse(content={"images": [], "message": "Aucune image pour le moment"}, status_code=200)

    # Initialize the image data list
    image_data = []

    # Check for images directly in the user's directory (outside categories)
    image_files = [f for f in os.listdir(user_dir) if os.path.isfile(user_dir / f)]
    for filename in image_files:
        image_url = f"http://localhost:8000/uploadSearch/{user_id}/{filename}"
        image_data.append({"url": image_url, "category": "uncategorized"})  # You can label them as "uncategorized" or similar

    # Get the categories subdirectories (i.e., each category has its own folder)
    categories = [category for category in os.listdir(user_dir) if os.path.isdir(user_dir / category)]

    if categories:
        for category in categories:
            category_dir = user_dir / category
            category_image_files = [f for f in os.listdir(category_dir) if os.path.isfile(category_dir / f)]
            for filename in category_image_files:
                image_url = f"http://localhost:8000/uploadSearch/{user_id}/{category}/{filename}"
                image_data.append({"url": image_url, "category": category})

    # Return the list of images
    return JSONResponse(content={"images": image_data})
@app.get("/getImages")
async def get_images(authorization: str = Header(None)):
    """
    Endpoint to get the images uploaded by the current user, with their category.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Check if user directory exists
    user_dir = UPLOAD_DIR3 / Path(user_id)

    if not user_dir.exists():
        # If directory doesn't exist, return no images
        return JSONResponse(content={"images": [], "message": "Aucune image pour le moment"}, status_code=200)

    # Get the categories subdirectories (i.e., each category has its own folder)
    categories = [category for category in os.listdir(user_dir) if os.path.isdir(user_dir / category)]

    if not categories:
        return JSONResponse(content={"images": [], "message": "Aucune image pour le moment"}, status_code=200)

    image_data = []

    for category in categories:
        category_dir = user_dir / category
        image_files = [f for f in os.listdir(category_dir) if os.path.isfile(category_dir / f)]

        for filename in image_files:
            image_url = f"http://localhost:8000/uploadSearch/{user_id}/{category}/{filename}"
            image_data.append({"url": image_url, "category": category})

    return JSONResponse(content={"images": image_data})


from pathlib import Path

UPLOAD_DIR3 = Path("uploadSearch")


async def save_file(file: UploadFile, user_id: str):
    user_dir = UPLOAD_DIR3 / Path(user_id)  # Correctly convert user_id to Path
    user_dir.mkdir(parents=True, exist_ok=True)
    file_location = user_dir / file.filename
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    return file_location


@app.get("/uploaded_images")
async def get_uploaded_images(
        folder_name: str = "uploadSearch",
        category: str = None,  # Added category parameter
        authorization: str = Header(None)
):
    try:
        # Verify that the Authorization header is present
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization token is missing.")
        # Extract the user ID from the JWT token
        token = authorization.split("Bearer ")[-1]
        user_id = decode_jwt(token)  # Implement `decode_jwt` to decode the JWT token

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")

        # Construct the user's folder path
        user_folder = os.path.join(folder_name, user_id)

        if not os.path.exists(user_folder):
            return JSONResponse(content={"message": "Aucune image téléchargée pour cet utilisateur."}, status_code=404)

        # Check if category is provided and filter images
        if category:
            category_folder = os.path.join(user_folder, category)
            if not os.path.exists(category_folder):
                return JSONResponse(content={"message": f"Aucune image dans la catégorie {category}."}, status_code=404)

            files = [f for f in os.listdir(category_folder) if os.path.isfile(os.path.join(category_folder, f))]
            image_urls = [f"/{folder_name}/{user_id}/{category}/{file}" for file in
                          files]  # Generate user-specific URLs
        else:
            # If no category, fetch all images
            files = [f for f in os.listdir(user_folder) if os.path.isfile(os.path.join(user_folder, f))]
            image_urls = [f"/{folder_name}/{user_id}/{file}" for file in files]

        return {"images": image_urls}
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)


@app.post("/upload_Search")
async def upload_file_for_search(file: UploadFile = File(...), category: str = Form(...),
                                 authorization: str = Header(None)):
    """
    Endpoint to upload a file and save it to a user-specific and category-specific directory.
    """
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Ensure category is provided
    if not category:
        raise HTTPException(status_code=400, detail="Category is missing")

    # Define the directory path based on user_id and category
    user_category_dir = f"uploadSearch/{user_id}/{category}"

    # Ensure the directory exists
    os.makedirs(user_category_dir, exist_ok=True)

    # Save the file in the appropriate directory
    file_location = os.path.join(user_category_dir, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    # Return the relative path of the uploaded file
    return {"file_path": f"/uploadSearch/{user_id}/{category}/{file.filename}"}


@app.post("/upload")
async def upload_file(file_path: str = Body(..., embed=True)):
    """
    Endpoint to process the file path and find similar images.
    """
    # Construct the full path for the uploaded image
    image_path = os.path.join(UPLOAD_DIR2, file_path.lstrip("/"))

    print(f"Received file path: {image_path}")

    # Check if the image exists at the specified path
    if not os.path.exists(image_path):
        return {"error": "Image not found."}

    # Get similar images based on the uploaded image
    similar_images = getSimilarImages(image_path)

    # Construct URLs for similar images
    image_urls = []
    for image in similar_images['ImagePath']:
        # First, replace backslashes with forward slashes using string.replace
        image_path_with_forward_slashes = image.replace('\\', '/')
        # Now format the URL using the modified image path
        image_url = f"http://127.0.0.1:8000/uploadSearch{image_path_with_forward_slashes}"
        image_urls.append(image_url)

    return {
        "image_path": image_path,  # Full image path for reference
        "similar_images": similar_images.to_dict(orient='records'),  # Format similar images as needed
        "ImagePath": image_urls  # URLs for similar images with forward slashes
    }


@app.post("/feedback")
async def update_weights(image_index: int = Body(..., embed=True)):
    # Load features and weights
    images_features = pd.read_csv(FEATURES_PATH)
    features = images_features.iloc[:, 2:].values  # Feature columns only (skip ImagePath and Category)
    weights = np.load(WEIGHTS_PATH)

    # Validate image index
    if image_index < 0 or image_index >= len(features):
        return {"error": "Image index out of range"}

    # Get the flagged image's features
    flagged_features = features[image_index]

    # Update weights for the specific image
    weights[image_index, :] = weights[image_index, :] * ALPHA + (1 - ALPHA) * flagged_features

    # Save the updated weights
    np.save(WEIGHTS_PATH, weights)

    return {"message": "Weights updated successfully"}


@app.get("/categories")
async def get_categories():
    images_features = pd.read_csv(FEATURES_PATH)
    categories = images_features['Category'].unique().tolist()
    return categories


@app.post("/categories/{category}")
async def get_images_by_categories(category: str):
    image_features = pd.read_csv(FEATURES_PATH)

    if category == "Tout" or not category:
        images_path_by_categories = image_features["ImagePath"].tolist()
    else:
        images_path_by_categories = image_features[image_features['Category'] == category]["ImagePath"].tolist()

    return {"images": images_path_by_categories}


@app.post("/apply_transformation")
async def apply_transformation(
        file: UploadFile = File(...),
        transformation: str = Form(...),
        authorization: str = Header(None)
):
    if authorization is None:
        raise HTTPException(status_code=400, detail="Authorization token is missing")

    # Extract token from Authorization header
    token = authorization.split("Bearer ")[-1]

    # Decode the token and get the user_id
    user_id = decode_jwt(token)
    try:
        # Charger l'image depuis le fichier uploadé
        image = Image.open(BytesIO(await file.read()))
        transformed_image = None

        # Appliquer la transformation
        if transformation == "crop":
            # Recadrer au centre
            width, height = image.size
            left = width // 4
            top = height // 4
            right = 3 * width // 4
            bottom = 3 * height // 4
            transformed_image = image.crop((left, top, right, bottom))
        elif transformation == "resize":
            # Redimensionner à 300x300
            transformed_image = image.resize((300, 300))
        elif transformation == "rotate":
            # Rotation de 90 degrés
            transformed_image = image.rotate(90)
        elif transformation == "blur":
            # Ajouter un effet de flou
            transformed_image = image.filter(ImageFilter.BLUR)
        else:
            raise HTTPException(status_code=400, detail="Transformation invalide")

        file_name, file_extension = os.path.splitext(file.filename)
        output_file_name = f"{file_name}_{transformation}{file_extension}"

        # Créer le dossier de l'utilisateur si nécessaire
        user_folder = os.path.join(UPLOAD_DIR2, user_id)
        os.makedirs(user_folder, exist_ok=True)

        # Enregistrer l'image transformée dans le dossier de l'utilisateur
        output_path = os.path.join(user_folder, output_file_name)
        transformed_image.save(output_path)

        # Retourner l'image transformée
        return FileResponse(output_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transformation : {str(e)}")


def apply_gabor_filter(image: np.ndarray) -> np.ndarray:
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply multiple Gabor filters with varied parameters
    filtered_images = []
    for theta in [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]:
        for sigma in [1.0, 2.0]:
            kernel = cv2.getGaborKernel(
                (31, 31),  # Larger kernel size
                sigma,  # Scale
                theta,  # Orientation
                10.0,  # Wavelength
                0.5,  # Aspect ratio
                0,  # Phase offset
                ktype=cv2.CV_32F
            )

            # Filter the image
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            filtered_images.append(filtered)

    # Combine filtered images
    combined = np.mean(filtered_images, axis=0)

    # Normalize and apply adaptive thresholding
    normalized = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(normalized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


@app.get("/characteristics/{idx}")
async def get_characteristics(idx: int):
    try:
        # Read the CSV file containing image features
        images_features = pd.read_csv(FEATURES_PATH)

        # Ensure idx is within bounds
        if idx < 0 or idx >= len(images_features):
            raise HTTPException(status_code=404, detail="Index out of range")

        # Get the image path from the DataFrame
        image_path = images_features.iloc[idx]["ImagePath"]
        image = cv2.imread("RSSCN7" + image_path)

        # Check if the image was loaded successfully
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Convert image to grayscale for Gabor filter application
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gabor filter to the image
        gabor_filtered_image = apply_gabor_filter(gray_image)

        # Convert the filtered image to a base64-encoded PNG image
        _, buffer = cv2.imencode('.png', gabor_filtered_image)
        gabor_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the binary Gabor-filtered image and the histogram data
        histograms = {}
        for i, color in enumerate(('b', 'g', 'r')):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten().tolist()
            histograms[color] = hist

        return {
            "histogram_colors": histograms,
            "gabor_image_base64": gabor_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
