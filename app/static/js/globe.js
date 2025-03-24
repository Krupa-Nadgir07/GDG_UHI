let selectedCoordinates = {};
let isSpinning = false;

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

require([
    "esri/Map",
    "esri/views/SceneView",
    "esri/geometry/Point"
], function (Map, SceneView, Point) {
    const map = new Map({
        basemap: "satellite",
        ground: "world-elevation"
    });

    const sceneView = new SceneView({
        container: "viewDiv",
        map: map,
        camera: {
            position: {
                latitude: 0,
                longitude: -90,
                z: 20000000 // Set initial zoom to show the globe
            },
            tilt: 0,
            heading: 0
        },
        environment: {
            background: {
                type: "color",
                color: [255, 255, 255, 1] // White background
            },
            starsEnabled: true,
            atmosphereEnabled: true
        },
        constraints: {
            altitude: {
                min: 100,
                max: 25000000
            }
        }
    });

    // Expose the SceneView globally
    window.sceneView = sceneView;

    let angle = 0;
    const radius = 20000000;
    const rotationDuration = 10000; // Spin for 10 seconds
    const rotationSpeed = 1.5; // Degrees per frame
    let rotationStartTime;

    function spinEarth(timestamp) {
        if (!rotationStartTime) rotationStartTime = timestamp;
        const elapsed = timestamp - rotationStartTime;

        if (elapsed < rotationDuration) {
            // Calculate the new angle for the rotation
            angle += rotationSpeed;
            if (angle >= 360) angle = 0;

            const radianAngle = (angle * Math.PI) / 180;
            const latitude = Math.sin(radianAngle) * 23.5;
            const longitude = angle - 90;

            // Update camera position
            sceneView.goTo({
                position: {
                    latitude: latitude,
                    longitude: longitude,
                    z: radius
                },
                heading: 180 - angle,
                tilt: 0
            }, { animate: false });

            requestAnimationFrame(spinEarth);
        } else {
            // Stop spinning and zoom into the selected location
            transitionTo2D();

        }
        delete selectedCoordinates;
    }

    async function transitionTo2D() {
        if (!selectedCoordinates.latitude || !selectedCoordinates.longitude) {
            console.error("Coordinates are not set. Cannot transition to 2D view.");
            return;
        }

        const targetPoint = new Point({
            latitude: parseFloat(selectedCoordinates.latitude),
            longitude: parseFloat(selectedCoordinates.longitude)
        });

        sceneView.goTo({
            target: targetPoint,
            scale: 50000, // Adjust for zoom level
            tilt: 0 // Add some tilt for perspective
        }, {
            duration: 5000, // Duration of zoom animation
            easing: "ease-in-out"
        }).then(() => {
            // After zooming in, adjust the layout
            document.getElementById("viewDiv").classList.remove("initial-globe");
            document.getElementById("viewDiv").classList.add("final-globe");

            setTimeout(function () {
                window.location.href = '/visualize'; // Redirect after 5 seconds
            }, 5000);
        });

        // wait(80000);
        // 
    }

    // Search Address Functionality
    async function fetchSearchResults(query) {
        const url = `https://geocode.maps.co/search?q=${encodeURIComponent(query)}&api_key=678284387ed0f830130830icr0e5e82`;

        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error("Failed to fetch geocoding data.");
            return await response.json();
        } catch (error) {
            alert(`Unable to fetch results. Please try again later. ${error.message}`);
            return [];
        }
    }

    function displayResults(results) {
        const dropdown = document.getElementById('resultsContainer');

        // Ensure the dropdown is visible
        dropdown.style.display = 'block'; // Ensure it's visible before appending results
        dropdown.innerHTML = ""; // Clear previous results

        if (results.length === 0) {
            const noResultItem = document.createElement("li");
            noResultItem.className = "mt-2 space-y-2 bg-white border border-gray-200 rounded-md shadow-lg max-h-48 overflow-y-auto";
            noResultItem.textContent = "No results found.";
            dropdown.appendChild(noResultItem);
            return;
        }

        results.forEach((result) => {
            const item = document.createElement('li');
            item.className = "mt-2 space-y-2 bg-white border border-gray-200 rounded-md shadow-lg max-h-48 overflow-y-auto";
            item.textContent = result.display_name;

            item.addEventListener('click', () => {
                // Set the selected coordinates
                selectedCoordinates = {
                    latitude: result.lat,
                    longitude: result.lon
                };

                // Clear dropdown
                dropdown.innerHTML = `<li class="mt-2 space-y-2 bg-white border border-gray-200 rounded-md shadow-lg max-h-48 overflow-y-auto">Searched location : ${result.display_name}</li>`;

                // Start spinning
                rotationStartTime = null; // Reset rotation time
                requestAnimationFrame(spinEarth);
                fetch('/location', {
                    method: 'POST',  // HTTP method
                    headers: {
                        'Content-Type': 'application/json',  // Specify that we're sending JSON
                    },
                    body: JSON.stringify(result)  // Convert data to JSON string
                })
                    .then(response => response.json())  // Parse JSON response from Flask
                    .then(data => {
                        console.log('Success:', data);
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
            });

            dropdown.appendChild(item);
        });
    }



    async function onButtonSearch() {
        const query = document.getElementById('searchInput').value.trim();
        if (query) {
            const results = await fetchSearchResults(query);

            displayResults(results);
        } else {
            alert("Please enter a search query.");
        }
    }

    // Expose searchAddress globally to use in the HTML
    // window.searchAddress = searchAddress;
    // window.onInputSearch = onInputSearch;
    window.onButtonSearch = onButtonSearch;
});