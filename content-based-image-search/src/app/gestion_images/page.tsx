"use client";

import { useEffect, useState } from "react";
import Navbar from "../components/navBar";
import axios from "@/lib/axios";
import {originConsoleError} from "next/dist/client/components/globals/intercept-console-error";
import {FaTrash} from "react-icons/fa";

export default function ImagesPage() {
  const [images, setImages] = useState<any[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [newCategoryName, setNewCategoryName] = useState<string>("");
  const [uploadedImages, setUploadedImages] = useState([]);
  const [imagesInFolder, setImagesInFolder] = useState([]);



  const [isLoading, setIsLoading] = useState<{
    categories: boolean;
    images: boolean;
    upload: boolean;
  }>({
    categories: true,
    images: false,
    upload: false,
  });

   const [error, setError] = useState<{
    categories?: string;
    images?: string;
    upload?: string;
  }>({});

  // Load categories from backend
  useEffect(() => {
    setIsLoading((prev) => ({ ...prev, categories: true }));
    setError((prev) => ({ ...prev, categories: undefined }));

    axios
      .get("/categories")
      .then((response) => {
        setCategories(response.data);
        setIsLoading((prev) => ({ ...prev, categories: false }));
      })
      .catch((error) => {
        console.error("Error fetching categories:", error);
        setError((prev) => ({
          ...prev,
          categories: error.response?.data?.message || "Impossible de charger les catégories",
        }));
        setIsLoading((prev) => ({ ...prev, categories: false }));
      });
  }, []);

  const handleFilter = (category: string) => {
    setSelectedCategory(category);
  };

  const handleAddCategory = () => {
    if (newCategoryName.trim() && !categories.includes(newCategoryName)) {
      setCategories((prevCategories) => [...prevCategories, newCategoryName]);
      setSelectedCategory(newCategoryName);
    }
    setNewCategoryName("");
    setIsModalOpen(false);
  };
const handleUpload = (event: React.FormEvent<HTMLFormElement>) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  const files = formData.getAll("image") as File[];
  const category = formData.get("category") as string;

  if (!category) {
    setError((prev) => ({ ...prev, upload: "Veuillez sélectionner une catégorie" }));
    return;
  }

  if (files.length === 0) {
    setError((prev) => ({ ...prev, upload: "Veuillez sélectionner au moins un fichier" }));
    return;
  }

  setIsLoading((prev) => ({ ...prev, upload: true }));
  setError((prev) => ({ ...prev, upload: undefined }));

  const uploadData = new FormData();
  files.forEach((file) => {
    uploadData.append("file", file);
  });
  uploadData.append("category", category);  // Ensure the category is appended here

  // Get the JWT token from sessionStorage
  const token = sessionStorage.getItem("jwtToken");
  console.log("Token:", token);

  if (!token) {
    setError((prev) => ({ ...prev, upload: "Utilisateur non authentifié" }));
    return;
  }

  // Send the files to the backend using axios.then() and catch()
  axios
    .post("/upload_Search", uploadData, {
      headers: {
        "Content-Type": "multipart/form-data",
        "Authorization": `Bearer ${token}`,  // Add the JWT token in the Authorization header
      },
    })
    .then((response) => {
      console.log("Response data:", response.data);

      if (response.status === 200) {
        // Add images directly to `uploadedImages` to display them immediately
        const uploadedImages = files.map((file) => ({
          id: crypto.randomUUID(),
          name: file.name,
          url: URL.createObjectURL(file), // Create a local URL for preview
          category: category,  // Use the selected category
        }));

        // Update state with the new images
        setUploadedImages((prev) => [...prev, ...uploadedImages]);
        setImagesInFolder((prev) => [...prev, ...uploadedImages]);

        // Call fetchUploadedImages after the upload to get the updated list of images from the backend
        fetchUploadedImages1();
      }
    })
    .catch((error) => {
      console.error("Error uploading images", error);
      setError((prev) => ({ ...prev, upload: "Une erreur s'est produite lors du téléchargement." }));
    })
    .finally(() => {
      setIsLoading((prev) => ({ ...prev, upload: false }));
    });
};








const fetchUploadedImages1 = async () => {
  try {
    const token = sessionStorage.getItem("jwtToken");

    if (!token) {
      setError((prev) => ({
        ...prev,
        images: "Utilisateur non authentifié",
      }));
      return;
    }

    const response = await axios.get("/getImages_for_search", {
      headers: {
        "Authorization": `Bearer ${token}`,  // Include the JWT token in the header
      },
    });

    const images = response.data.images || [];
    console.log("IMAGES FROM BACKEND:", images);

    setUploadedImages(images);
    setImagesInFolder(images);
  } catch (error) {
    setError((prev) => ({ ...prev, images: "Aucune image pour le moment" }));
  }
};


useEffect(() => {
  const fetchUploadedImages = async () => {
    try {
      // Get the JWT token from sessionStorage
      const token = sessionStorage.getItem("jwtToken");

      // Check if the token exists
      if (!token) {
        setError((prev) => ({
          ...prev,
          images: "Utilisateur non authentifié",
        }));
        return;
      }

      // Make the request with the Authorization header
      const response = await axios.get("/getImages_for_search", {
        headers: {
          "Authorization": `Bearer ${token}`, // Include the JWT token in the Authorization header
        },
      });

      // Extract the images from the response
      const images = response.data.images || [];

      // Check if there are no images
      if (images.length === 0) {
        setError((prev) => ({
          ...prev,
          images: "Aucune image pour le moment", // Message for no images
        }));
      } else {
        // Update the state with the fetched images
        setUploadedImages(images);
        setImagesInFolder(images);
      }

      console.log("Uploaded images:", images);
    } catch (error) {
      // Handle errors if they occur
      setError((prev) => ({
        ...prev,
        images: "Erreur lors du chargement ades images.",
      }));
    }
  };

  fetchUploadedImages(); // Call the fetch function when the component mounts

}, []);  // Empty dependency array so this effect runs only once when the component mounts

  // Filter images by category
  const filteredImages =
    selectedCategory === ""
      ? uploadedImages
      : uploadedImages.filter((image) => image.category === selectedCategory);

  const handleDeleteImage = async (imageUrl) => {
  const token = sessionStorage.getItem("jwtToken");

  if (!token) {
    setError((prev) => ({
      ...prev,
      images: "Utilisateur non authentifié",
    }));
    return;
  }

  try {
    // Send a request to delete the image
    await axios.delete("/deleteImage", {
      headers: {
        "Authorization": `Bearer ${token}`, // Include the JWT token in the Authorization header
      },
      params: {
        image_url: imageUrl,
      },
    });

    // Update the state to remove the image from the list
    setUploadedImages((prevImages) =>
      prevImages.filter((image) => image.url !== imageUrl)
    );
    alert("Image deleted successfully");
  } catch (error) {
    console.error("Error deleting image:", error);
    alert("Erreur lors de la suppression de l'image.");
  }
};

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar activeSection="images" />
      <div className="flex-1 py-8 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Upload Section */}
          <div className="bg-white p-6 rounded-xl shadow-lg mb-10">
            <h2 className="text-3xl font-semibold text-gray-800 mb-6">Uploader des Images</h2>

            {/* Error Message for Upload */}
            {error.upload && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
                {error.upload}
              </div>
            )}

            <form onSubmit={handleUpload} className="flex space-x-4">
              <input
                type="file"
                name="image"
                multiple
                disabled={isLoading.upload}
                className="flex-1 py-3 px-5 bg-gray-100 rounded-md border border-gray-300 shadow-sm text-black disabled:opacity-50"
              />
              <select
                name="category"
                disabled={isLoading.categories || isLoading.upload}
                className="py-3 px-5 bg-gray-100 rounded-md border border-gray-300 shadow-sm text-black disabled:opacity-50"
                value={selectedCategory}
                onChange={(e) => {
                  const value = e.target.value;
                  value === "add-new-category"
                    ? setIsModalOpen(true)
                    : setSelectedCategory(value);
                }}
              >
                <option value="" disabled>
                  {isLoading.categories ? "Chargement..." : "Choisir une catégorie"}
                </option>
                {categories.map((category, index) => (
                  <option key={index} value={category}>
                    {category}
                  </option>
                ))}
                <option value="add-new-category">Ajouter une nouvelle catégorie</option>
              </select>
              <button
                type="submit"
                disabled={isLoading.upload || isLoading.categories}
                className={`px-6 py-3 rounded-md transition-all duration-300 ${
                  isLoading.upload || isLoading.categories
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-indigo-600 text-white hover:bg-indigo-700"
                }`}
              >
                {isLoading.upload ? "Téléchargement..." : "Uploader"}
              </button>
            </form>

            {/* Categories Loading Error */}
            {error.categories && (
              <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative mt-4" role="alert">
                {error.categories}
              </div>
            )}
          </div>

          {/* Category Buttons */}
          <div className="flex space-x-4 mb-8">
            <button
              onClick={() => handleFilter("")}
              disabled={isLoading.images}
              className={`px-6 py-3 rounded-md ${
                selectedCategory === ""
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-200 text-gray-700"
              } disabled:opacity-50`}
            >
              Tout
            </button>
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => handleFilter(category)}
                disabled={isLoading.images}
                className={`px-6 py-3 rounded-md ${
                  selectedCategory === category
                    ? "bg-indigo-600 text-white"
                    : "bg-gray-200 text-gray-700"
                } disabled:opacity-50`}
              >
                {category}
              </button>
            ))}
          </div>

          {/* Images Loading State */}
          {isLoading.images && (
            <div className="flex justify-center items-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-blue-500"></div>
            </div>
          )}

          {/* Images Error State */}
          {error.images && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
              {error.images}
            </div>
          )}

          {/* Image Display */}
          {!isLoading.images && !error.images && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {filteredImages.length > 0 ? (
                    filteredImages.map((image, index) => (
                        <div key={index} className="bg-white rounded-xl shadow-lg overflow-hidden">
                          <div className="relative">
                            <img
                                src={image.url}
                                alt={`Image ${index + 1}`}
                                loading="lazy"
                                className="w-full h-56 object-cover transition-transform duration-300 hover:scale-105"
                            />
                            {/* Delete Button - Positioned to the Right */}
                            <button
                                onClick={() => handleDeleteImage(image.url)}
                                className="absolute top-2 right-2 p-2 bg-white rounded-full shadow-md hover:bg-gray-100 transition-all"
                                style={{background: 'none', border: 'none', cursor: 'pointer'}}
                            >
                              <FaTrash color="red" size={20}/> {/* Delete icon */}
                            </button>
                          </div>
                          <div className="p-4">
                            <h3 className="text-lg font-semibold text-gray-800">Image {index + 1}</h3>
                            <p className="text-sm text-gray-600">{image.category}</p>
                          </div>
                        </div>
                    ))
                ) : (
                    <div className="col-span-full text-center text-gray-500">
                      Aucune image trouvée
                    </div>
                )}
              </div>

          )}
        </div>
      </div>

      {/* Add New Category Modal */}
      {isModalOpen && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="bg-white p-6 rounded-xl shadow-lg w-96">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">Ajouter une Nouvelle Catégorie</h3>
              <input
                  type="text"
                  className="w-full py-3 px-5 mb-4 bg-gray-100 rounded-md border border-gray-300"
                  placeholder="Nom de la catégorie"
                  value={newCategoryName}
                  onChange={(e) => setNewCategoryName(e.target.value)}
              />
              <div className="flex justify-end space-x-4">
                <button
                    onClick={() => setIsModalOpen(false)}
                    className="px-5 py-2 bg-gray-500 rounded-md hover:bg-gray-400"
                >
                  Annuler
                </button>
                <button
                    onClick={handleAddCategory}
                    className="px-5 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                >
                  Ajouter
                </button>
              </div>
            </div>
          </div>
      )}
    </div>
  );
}