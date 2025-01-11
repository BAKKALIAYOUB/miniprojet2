"use client";
import { useState, useEffect } from "react";
import { Chart, BarController, BarElement, LinearScale, CategoryScale, Title, Legend } from 'chart.js/auto';
import Navbar from "../components/navBar";
import axios from "@/lib/axios";

Chart.register(BarController, BarElement, LinearScale, CategoryScale, Title, Legend);

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [activeSection, setActiveSection] = useState("search");
  const [uploadedImage, setUploadedImage] = useState(null);
  const [results, setResults] = useState([]);
  const [sim_images, set_sim_images] = useState([]);
  const [desablerecherhcemethode, setDesablerecherhcemethode] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);
  const [imagesInFolder, setImagesInFolder] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gaborImage, setGaborImage] = useState(null);
  const [reduceMesh, setReduceMesh] = useState(false);


  const extractImageName = (url) => {
    return url.split("/").pop(); // Extracts the last part of the URL (e.g., "y5160_clay_vase.jpg")
  };

  const handleSelectImage = (image) => {
    const previewUrl = `${image.url}`;
    setUploadedImage({ file: image, preview: previewUrl });
    setIsModalOpen(false);

    // Extract and log the image name
    const imageName = extractImageName(image.url);
    console.log("Selected Image Name:", imageName);
  };

  useEffect(() => {
    // Fetch uploaded images for the current user
    const fetchUploadedImages = async () => {
      try {
        // Retrieve the JWT token from sessionStorage
        const token = sessionStorage.getItem("jwtToken");

        if (!token) {
          setError("Utilisateur non authentifié.");
          return;
        }

        // Make a request to the server with the user's token
        const response = await axios.get("/getImages_for_search", {
          headers: {
            Authorization: `Bearer ${token}` // Include the token in the Authorization header
          },
        });

        // Update the state with the fetched images
        const images = response.data.images || [];
        console.log("IMAGES FROM BACKEND:", images);
        setUploadedImages(images);
        setImagesInFolder(images);
      } catch (error) {
        // Handle errors
        setError(
          error.response?.data?.detail || "Erreur lors du chargement des images."
        );
      }
    };

    fetchUploadedImages(); // Fetch images when the component mounts
  }, []);

  useEffect(() => {
    return () => {
      if (uploadedImage) {
        URL.revokeObjectURL(uploadedImage.preview);
      }
    };
  }, [uploadedImage]);

  const handleSearch = async () => {
    setError(null);
    setResults([]);

    if (!uploadedImage?.file) {
      setError("Veuillez sélectionner une image.");
      return;
    }

    try {
      setIsLoading(true);
      setDesablerecherhcemethode(true);

      const url = uploadedImage.file.url;
      const path = url.replace("http://localhost:8000/uploadSearch", ""); // Keep the `/uploadSearch` part

      const response = await axios.post("/upload", { file_path: path, reduce_mesh: reduceMesh });

      const similarImages = response.data.similar_images || [];
      console.log("hi", response.data);
      setResults(similarImages);
      const images_path = response.data.ImagePath
      console.log(images_path)
      set_sim_images(images_path)

      if (similarImages.length === 0) {
        setError("Aucune image similaire trouvée.");
      }
    } catch (error) {
      setError("Une erreur s'est produite pendant l'upload de l'image. Veuillez réessayer.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setDesablerecherhcemethode(false);
    setResults([]);
    setError(null);
  };

  const handleIrrelevantChange = async (index_of_irrelevant_image) => {
    try {
      setIsLoading(true);
      await axios.post("/feedback", { image_index: index_of_irrelevant_image });
      await handleSearch();
    } catch (error) {
      setError("Une erreur s'est produite lors de la mise à jour des poids.");
    } finally {
      setIsLoading(false);
    }
  };

  const openImageModal = (image) => {
    setSelectedImage(image);
    setIsImageModalOpen(true);

    axios.get(`/characteristics/${image.index}`)
      .then((response) => {
        console.log(response.data)
        const histogram = response.data.histogram_colors;
        setGaborImage(response.data.gabor_image_base64)
        renderHistColor(histogram);
      });
  };

  const renderHistColor = (histColor) => {
    const canvas = document.getElementById("histogramColorChart");

    if (canvas && histColor) {
      const existingChart = Chart.getChart(canvas);
      if (existingChart) {
        existingChart.destroy();
      }

      const labels = Array.from({ length: 256 }, (_, i) => i);

      new Chart(canvas, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Red",
              data: histColor.r,
              backgroundColor: "rgba(255, 99, 132, 0.5)",
              borderColor: "rgba(255, 99, 132, 1)",
              borderWidth: 1,
            },
            {
              label: "Green",
              data: histColor.g,
              backgroundColor: "rgba(75, 192, 192, 0.5)",
              borderColor: "rgba(75, 192, 192, 1)",
              borderWidth: 1,
            },
            {
              label: "Blue",
              data: histColor.b,
              backgroundColor: "rgba(54, 162, 235, 0.5)",
              borderColor: "rgba(54, 162, 235, 1)",
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: { display: true },
          },
          scales: {
            x: {
              beginAtZero: true,
              title: {
                display: true,
                text: "Intensity (0-255)",
              },
            },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: "Frequency",
              },
            },
          },
        },
      });
    }
  };

  const closeImageModal = () => {
    setIsImageModalOpen(false);
    setSelectedImage(null);
  };

  const sections = {
    search: (
      <div>
        <div className="container mx-auto px-6 py-6">
          <div className="flex flex-col space-y-4">
            <div className="flex items-center space-x-4">
              <button
                  onClick={() => setIsModalOpen(true)}
                  className="px-4 py-2 border rounded-md text-black"
                  disabled={isLoading}
              >
                Choisir une image
              </button>
              <div className="flex items-center space-x-2">
                <input
                    type="checkbox"
                    id="reduceMesh"
                    checked={reduceMesh}
                    onChange={() => setReduceMesh(!reduceMesh)} // Toggle the state
                    className="form-checkbox"
                />
                <label htmlFor="reduceMesh" className="text-black">Réduire le maillage</label>
              </div>
              <button
                  onClick={handleSearch}
                  className={`px-4 py-2 rounded-md ${
                      isLoading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-500 text-white hover:bg-blue-600"
                  }`}
                  disabled={isLoading}
              >
                {isLoading ? "Recherche en cours..." : "Rechercher"}
              </button>
              <button
                  className="px-4 py-2 text-white rounded-md bg-blue-500 hover:bg-blue-600"
                  disabled={!desablerecherhcemethode}
                  onClick={handleReset}
              >
                Clear
              </button>
            </div>

            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                  <span className="block sm:inline">{error}</span>
                </div>
            )}

            {uploadedImage && (
                <div className="mt-4 flex justify-center items-center">
                  <div className="text-center">
                  <h3 className="text-lg font-semibold mb-2 text-black">Image Requête:</h3>
                  <img
                    src={uploadedImage.preview}
                    alt="Image Requête"
                    className="w-80 h-80 object-cover mx-auto rounded-md shadow-lg"
                  />
                </div>
              </div>
            )}
          </div>
          {/* Resulat de recherche */}
          <div className="mt-6">
            <h2 className="text-xl font-semibold">Résultats de la Recherche</h2>
            <div className="grid grid-cols-3 gap-4 mt-4">
              {!isLoading && results.length > 0 && (
                results.map((image, idx) => {
                  // Use the corresponding image URL from sim_images
                  const imageUrl = sim_images[idx];
                  return (
                    <div key={idx} className="bg-white shadow-lg rounded-md p-4">
                      <img
                        src={encodeURI(imageUrl)} // Encode the URL to handle spaces
                        alt={image.title || "Image"}
                        className="w-full h-80 object-cover rounded-md cursor-pointer"
                        onClick={() => openImageModal(image)
                        }
                      />
                      <h3 className="text-center mt-2 text-black">{image.title || "Untitled"}</h3>
                      <h2 className="text-center mt-2">{image.distance || "N/A"}</h2>
                      <h2 className="text-center mt-2">{image.Category || "Unknown"}</h2>
                      <button
                        className="p-2 border rounded bg-blue-600 text-white hover:bg-blue-500"
                        onClick={() => {
                          // Construct the download URL
                          const downloadUrl = `http://127.0.0.1:8000/download/${imageUrl}`;

                          // Create a temporary link element
                          const link = document.createElement('a');
                          link.href = downloadUrl;  // Set the link's href to the download URL
                          link.download = imageUrl.split("/").pop(); // Set the download filename (same as image name)

                          // Trigger the click to download the file
                          link.click();
                        }}
                      >
                        Download
                      </button>
                    </div>
                  );
                })
              )}

              {!isLoading && results.length === 0 && (
                  <div className="col-span-3 text-center text-gray-600">
                    Aucun résultat trouvé.
                  </div>
              )}
            </div>
          </div>
        </div>
      </div>
    ),
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar activeSection={activeSection} setActiveSection={setActiveSection} />
      <div className="flex-1 bg-gray-50 py-6">
        <div className="container mx-auto px-6">{sections[activeSection]}</div>
      </div>

      {/* Images Uploaded by the user */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-white p-6 rounded-md shadow-lg w-1/2">
            <h2 className="text-xl font-semibold mb-4">Choisissez une image</h2>
            <div className="grid grid-cols-3 gap-4">
              {imagesInFolder.map((image, idx) => (
                <div key={idx} className="cursor-pointer" onClick={() => handleSelectImage(image)}>
                  <img
                    src={`${image.url}`}
                    alt={`Image ${idx}`}
                    className="w-full h-32 object-cover rounded-md"
                  />
                </div>
              ))}
            </div>
            <button
              onClick={() => setIsModalOpen(false)}
              className="mt-4 px-4 py-2 bg-gray-600 text-white rounded-md"
            >
              Fermer
            </button>
          </div>
        </div>
      )}

      {/* Display of image Caracterstics */}
      {isImageModalOpen && selectedImage && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex justify-center items-center z-50">
          <div className="bg-white p-6 rounded-md shadow-lg w-7xl max-w-7xl">
            <div className="mb-6">
              <div className="flex space-x-4">
                <div className="flex-1">
                  <h2 className="text-xl font-semibold mb-4 text-center">Histogramme des couleurs</h2>
                  <canvas id="histogramColorChart" className="w-full h-96"></canvas>
                </div>
                <div className="flex-1">
                  {gaborImage ? (
                      <div>
                        <h2 className="text-xl font-semibold mb-4 text-center">Gabor Filtered Image</h2>
                        <img
                            className={"w-full h-64 object-cover"}
                            src={`data:image/png;base64,${gaborImage}`}
                            alt="Gabor Filtered Image"
                        />
                      </div>
                  ) : (
                      <></>
                  )}
                </div>
              </div>
            </div>
            <button
                onClick={closeImageModal}
                className="mt-4 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}