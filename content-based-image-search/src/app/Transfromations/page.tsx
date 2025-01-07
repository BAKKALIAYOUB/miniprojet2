"use client";

import { useState, useEffect } from "react";
import Navbar from "../components/navBar";
import axios from "@/lib/axios";
import { FiImage } from "react-icons/fi";

export default function App() {
  const [activeSection, setActiveSection] = useState("transformations");
  const [uploadedImages, setUploadedImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [transformation, setTransformation] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [resultImage, setResultImage] = useState(null);  // Ajout de l'état resultImage

useEffect(() => {
  const fetchUploadedImages = async () => {
    try {
      // Get the JWT token from sessionStorage
      const token = sessionStorage.getItem("jwtToken");

      // Check if the token exists
      if (!token) {
        setError("Utilisateur non authentifié.");
        return;
      }

      // Make the request with the Authorization header
      const response = await axios.get("/getImages ", {
        headers: {
          Authorization: `Bearer ${token}`, // Include the JWT token
        },
      });

      // Set images or handle the case of no images
      const images = response.data.images || [];

      if (images.length === 0) {
        setError("Aucune image pour le moment.");
      } else {
        setUploadedImages(images);
      }
    } catch (error) {
      // Handle errors if they occur
      setError("Erreur lors du chargement des images.");
    }
  };

  fetchUploadedImages(); // Call the fetch function when the component mounts
}, []); // Empty dependency array so this effect runs only once when the component mounts


  const handleSelectImage = (image) => {
    setSelectedImage(image);
    console.log("hiii",image)
    setResultImage(null);  // Réinitialiser l'image transformée
    setError(null);
    setIsModalOpen(false);
  };

const applyTransformation = async () => {
  if (!selectedImage || !transformation) {
    setError("Veuillez sélectionner une image et une transformation.");
    return;
  }

  try {
    setIsLoading(true);
    setError(null);

    // Assuming userId is available (e.g., from session or context)
    const userId = sessionStorage.getItem("jwtToken");  // You can get this from the user's session

    if (!userId) {
      setError("Utilisateur non authentifié.");
      return;
    }

    console.log("Image:", selectedImage);

    // Load the image from the server (fetch as blob)
    const imageResponse = await axios.get(selectedImage.url.replace('http://localhost:8000', ''), {
      responseType: 'blob'
    });
    console.log("Fetched Image:", imageResponse);

    // Prepare FormData
    const formData = new FormData();
    formData.append("file", imageResponse.data, selectedImage.url.split('/').pop()); // Ensure filename is set correctly
    formData.append("transformation", transformation);
    formData.append("user_id", userId);  // Pass the user ID to the backend

    console.log("FormData:",selectedImage.url);

    // Send the image to the backend for transformation
    const transformResponse = await axios.post("/apply_transformation", formData, {
      headers: { "Content-Type": "multipart/form-data","Authorization": `Bearer ${sessionStorage.getItem("jwtToken")}` },

      responseType: "blob",
    });

    // Create a URL for the transformed image
    const blobUrl = URL.createObjectURL(new Blob([transformResponse.data]));
    setResultImage(blobUrl);  // Set the transformed image URL
  } catch (error) {
    setError("Une erreur s'est produite pendant la transformation.");
    console.error(error);
  } finally {
    setIsLoading(false);
  }
};






  return (
    <div className="min-h-screen flex flex-col">
      <Navbar activeSection={activeSection} setActiveSection={setActiveSection} />
      <div className="flex-1 bg-gray-50 py-6">
        <div className="container mx-auto px-6 space-y-6">
          <h2 className="text-2xl font-semibold text-center">Transformations d'Images</h2>

          <div className="flex items-center justify-center space-x-4 mt-6">
            <button
              onClick={() => setIsModalOpen(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 shadow-md"
            >
              Choisir une image
            </button>
            <select
              value={transformation}
              onChange={(e) => setTransformation(e.target.value)}
              className="block w-60 text-sm text-gray-700 border border-gray-300 rounded-lg p-2"
            >
              <option value="">Sélectionner une transformation</option>
              <option value="crop">Recadrer</option>
              <option value="resize">Redimensionner</option>
              <option value="rotate">Rotation</option>
              <option value="blur">Flou</option>
            </select>
            <button
              onClick={applyTransformation}
              className={`flex items-center space-x-2 px-4 py-2 rounded-full text-white ${
                isLoading || !selectedImage ? "bg-gray-400 cursor-not-allowed" : "bg-green-500 hover:bg-green-600"
              }`}
              disabled={isLoading || !selectedImage}
            >
              <FiImage />
              <span>{isLoading ? "Application..." : "Appliquer"}</span>
            </button>
          </div>

          {isModalOpen && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white p-6 rounded-lg w-3/4 max-h-[80%] overflow-y-auto shadow-lg">
                <h3 className="text-lg font-medium mb-4">Sélectionner une image</h3>
                <div className="grid grid-cols-3 gap-4">
                  {uploadedImages.map((image, index) => (
                    <div
                      key={index}
                      className="p-2 border rounded-lg cursor-pointer hover:border-blue-500"
                      onClick={() => handleSelectImage(image)}
                    >
                      <img
                        src={`${image.url}`}
                        alt="Image"
                        className="w-full h-32 object-cover"
                      />
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="mt-4 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
                >
                  Fermer
                </button>
              </div>
            </div>
          )}

          {error && <div className="text-red-500 text-center mt-4">{error}</div>}

          <div className="flex justify-center space-x-4 mt-6">
            {selectedImage && !error && (
              <div className="text-center">
                <h3 className="text-lg font-medium mb-4">Image originale</h3>
                <img
                  src={`${selectedImage.url}`}
                  alt="Image originale"
                  className="max-w-md mx-auto shadow-lg"
                />
              </div>
            )}

            {resultImage && !error && (
              <div className="text-center">
                <h3 className="text-lg font-medium mb-4">Image transformée</h3>
                <img
                  src={resultImage}
                  alt="Image transformée"
                  className="max-w-md mx-auto shadow-lg"
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}