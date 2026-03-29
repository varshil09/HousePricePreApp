document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const priceSpan = document.getElementById('price');
    const rangeSpan = document.getElementById('range');
    const errorMessageSpan = document.getElementById('errorMessage');
    const predictBtn = document.querySelector('.predict-btn');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Hide previous results
        resultDiv.style.display = 'none';
        errorDiv.style.display = 'none';

        // Get form data - only the key features
        const formData = {
            // Size & Area
            GrLivArea: parseInt(document.getElementById('GrLivArea').value),
            LotArea: parseInt(document.getElementById('LotArea').value),
            TotalBsmtSF: parseInt(document.getElementById('TotalBsmtSF').value),
            '1stFlrSF': parseInt(document.getElementById('1stFlrSF').value),

            // Quality & Age
            OverallQual: parseInt(document.getElementById('OverallQual').value),
            OverallCond: parseInt(document.getElementById('OverallCond').value),
            YearBuilt: parseInt(document.getElementById('YearBuilt').value),
            YearRemodAdd: parseInt(document.getElementById('YearRemodAdd').value),

            // Rooms
            BedroomAbvGr: parseInt(document.getElementById('BedroomAbvGr').value),
            FullBath: parseInt(document.getElementById('FullBath').value),
            HalfBath: parseInt(document.getElementById('HalfBath').value),
            Fireplaces: parseInt(document.getElementById('Fireplaces').value),

            // Garage
            GarageCars: parseInt(document.getElementById('GarageCars').value),
            GarageArea: parseInt(document.getElementById('GarageArea').value),

            // Location
            Neighborhood: document.getElementById('Neighborhood').value,
            MSZoning: document.getElementById('MSZoning').value,

            // Outdoor
            WoodDeckSF: parseInt(document.getElementById('WoodDeckSF').value),
            OpenPorchSF: parseInt(document.getElementById('OpenPorchSF').value)
        };

        // Add computed fields that the model might expect
        formData['2ndFlrSF'] = 0;  // Default value
        formData['KitchenAbvGr'] = 1;  // Most houses have 1 kitchen
        formData['TotRmsAbvGrd'] = formData.BedroomAbvGr + formData.FullBath + 3;  // Approximate
        formData['LotFrontage'] = Math.sqrt(formData.LotArea) / 2;  // Approximate frontage

        // Basic validation
        if (formData.GrLivArea < 300) {
            showError('Living area must be at least 300 sq ft');
            return;
        }

        if (formData.OverallQual < 1 || formData.OverallQual > 10) {
            showError('Overall quality must be between 1 and 10');
            return;
        }

        // Set loading state
        setLoading(true);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.success) {
                priceSpan.textContent = data.formatted_price;

                if (data.confidence_range) {
                    rangeSpan.innerHTML =
                        `${data.confidence_range.formatted_lower} - ${data.confidence_range.formatted_upper}`;
                }

                resultDiv.style.display = 'block';

                // Add animation
                priceSpan.classList.add('pulse');
                setTimeout(() => priceSpan.classList.remove('pulse'), 1000);

                // Scroll to result
                resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                showError(data.error || 'Prediction failed');
            }
        } catch (error) {
            showError('Cannot connect to server. Make sure Flask is running on port 5000');
            console.error(error);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            predictBtn.innerHTML = '<span class="loading"></span> Predicting...';
            predictBtn.disabled = true;
        } else {
            predictBtn.innerHTML = 'Predict Price';
            predictBtn.disabled = false;
        }
    }

    function showError(message) {
        errorMessageSpan.textContent = message;
        errorDiv.style.display = 'block';

        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }

    // Input validation
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            let value = parseFloat(this.value);

            if (isNaN(value)) value = min;
            if (value < min) this.value = min;
            if (value > max) this.value = max;
        });
    });

    // Auto-populate YearRemodAdd if empty
    const yearBuilt = document.getElementById('YearBuilt');
    const yearRemod = document.getElementById('YearRemodAdd');

    yearBuilt.addEventListener('change', function() {
        if (!yearRemod.value || yearRemod.value < this.value) {
            yearRemod.value = this.value;
        }
    });

    // Tooltips for better UX
    const tooltips = {
        'GrLivArea': 'Total living area above ground',
        'OverallQual': 'Overall material and finish quality',
        'Neighborhood': 'Location within Ames'
    };

    Object.keys(tooltips).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.setAttribute('title', tooltips[id]);
        }
    });
});