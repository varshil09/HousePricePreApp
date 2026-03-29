// Global chart instances
let charts = {};

// Tab switching function
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Add active class to clicked button
    event.target.classList.add('active');

    // Refresh charts when switching to analytics tab
    if (tabName === 'analytics') {
        setTimeout(() => {
            refreshAllCharts();
        }, 100);
    }

    // Load insights when switching to insights tab
    if (tabName === 'insights') {
        loadInsights();
    }
}

// Refresh all charts
async function refreshAllCharts() {
    await loadStatistics();
    await loadPriceDistribution();
    await loadCorrelationHeatmap();
    await loadNeighborhoodChart();
    await loadYearTrendChart();
    await loadQualityChart();
    await loadFeatureImportance();
    await loadScatterPlot('GrLivArea');
}

// Load statistics
async function loadStatistics() {
    try {
        const response = await fetch('http://localhost:5000/api/statistics');
        const data = await response.json();

        document.getElementById('avg-price').textContent = `$${data.mean_price.toLocaleString()}`;
        document.getElementById('median-price').textContent = `$${data.median_price.toLocaleString()}`;
        document.getElementById('price-range').textContent = `$${data.min_price.toLocaleString()} - $${data.max_price.toLocaleString()}`;
        document.getElementById('total-houses').textContent = data.total_houses.toLocaleString();

        // Update prediction card stats
        document.getElementById('avg-living-area').textContent = Math.round(data.avg_living_area).toLocaleString();
        document.getElementById('avg-lot-area').textContent = Math.round(data.avg_lot_area).toLocaleString();
        document.getElementById('avg-quality').textContent = data.avg_qual_rating.toFixed(1);
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Load price distribution histogram
async function loadPriceDistribution() {
    try {
        const response = await fetch('http://localhost:5000/api/price-distribution');
        const data = await response.json();

        const ctx = document.getElementById('priceHistogram').getContext('2d');

        // Create histogram bins
        const minPrice = Math.min(...data.prices);
        const maxPrice = Math.max(...data.prices);
        const binCount = 30;
        const binWidth = (maxPrice - minPrice) / binCount;

        const bins = Array(binCount).fill(0);
        data.prices.forEach(price => {
            const binIndex = Math.min(Math.floor((price - minPrice) / binWidth), binCount - 1);
            bins[binIndex]++;
        });

        const labels = Array(binCount).fill().map((_, i) => {
            const start = minPrice + i * binWidth;
            const end = start + binWidth;
            return `$${Math.round(start/1000)}k-$${Math.round(end/1000)}k`;
        });

        if (charts.priceHistogram) charts.priceHistogram.destroy();

        charts.priceHistogram = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of Houses',
                    data: bins,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: { callbacks: { label: (ctx) => `${ctx.raw} houses` } }
                },
                scales: {
                    y: { title: { display: true, text: 'Frequency' } },
                    x: { title: { display: true, text: 'Price Range' }, ticks: { rotation: 45 } }
                }
            }
        });
    } catch (error) {
        console.error('Error loading price distribution:', error);
    }
}

// Load correlation heatmap
async function loadCorrelationHeatmap() {
    try {
        const response = await fetch('http://localhost:5000/api/feature-correlation');
        const data = await response.json();

        const ctx = document.getElementById('correlationHeatmap').getContext('2d');

        // Prepare data for heatmap (using a custom chart)
        const features = data.features;
        const matrix = data.matrix;

        // Create dataset for heatmap
        const heatmapData = [];
        for (let i = 0; i < features.length; i++) {
            for (let j = 0; j < features.length; j++) {
                heatmapData.push({
                    x: j,
                    y: i,
                    v: matrix[i][j]
                });
            }
        }

        if (charts.correlationHeatmap) charts.correlationHeatmap.destroy();

        charts.correlationHeatmap = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Correlation',
                    data: heatmapData,
                    backgroundColor: heatmapData.map(d => {
                        const value = Math.abs(d.v);
                        if (value > 0.7) return 'rgba(255, 99, 132, 0.8)';
                        if (value > 0.4) return 'rgba(255, 159, 64, 0.8)';
                        if (value > 0.2) return 'rgba(255, 205, 86, 0.8)';
                        return 'rgba(75, 192, 192, 0.8)';
                    }),
                    pointRadius: 20,
                    pointHoverRadius: 25
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const dataPoint = context.raw;
                                return `${features[dataPoint.y]} vs ${features[dataPoint.x]}: ${dataPoint.v.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Features' },
                        ticks: { callback: (val) => features[val], autoSkip: false, rotation: 45 }
                    },
                    y: {
                        title: { display: true, text: 'Features' },
                        ticks: { callback: (val) => features[val] }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading correlation heatmap:', error);
    }
}

// Load neighborhood chart
async function loadNeighborhoodChart() {
    try {
        const response = await fetch('http://localhost:5000/api/price-by-neighborhood');
        const data = await response.json();

        const ctx = document.getElementById('neighborhoodChart').getContext('2d');

        // Take top 10 neighborhoods
        const topNeighborhoods = data.neighborhoods.slice(0, 10);
        const topPrices = data.prices.slice(0, 10);

        if (charts.neighborhoodChart) charts.neighborhoodChart.destroy();

        charts.neighborhoodChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: topNeighborhoods,
                datasets: [{
                    label: 'Average Price',
                    data: topPrices,
                    backgroundColor: 'rgba(153, 102, 255, 0.7)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { position: 'top' },
                    tooltip: { callbacks: { label: (ctx) => `$${ctx.raw.toLocaleString()}` } }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Price ($)' },
                        ticks: { callback: (val) => `$${val.toLocaleString()}` }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading neighborhood chart:', error);
    }
}

// Load year trend chart
async function loadYearTrendChart() {
    try {
        const response = await fetch('http://localhost:5000/api/price-over-time');
        const data = await response.json();

        const ctx = document.getElementById('yearTrendChart').getContext('2d');

        if (charts.yearTrendChart) charts.yearTrendChart.destroy();

        charts.yearTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.years,
                datasets: [{
                    label: 'Average Price',
                    data: data.prices,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: { callbacks: { label: (ctx) => `$${ctx.raw.toLocaleString()}` } }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Price ($)' },
                        ticks: { callback: (val) => `$${val.toLocaleString()}` }
                    },
                    x: { title: { display: true, text: 'Year Built' } }
                }
            }
        });
    } catch (error) {
        console.error('Error loading year trend chart:', error);
    }
}

// Load quality chart
async function loadQualityChart() {
    try {
        const response = await fetch('http://localhost:5000/api/price-by-quality');
        const data = await response.json();

        const ctx = document.getElementById('qualityChart').getContext('2d');

        if (charts.qualityChart) charts.qualityChart.destroy();

        charts.qualityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.qualities,
                datasets: [
                    {
                        label: 'Average Price',
                        data: data.means,
                        backgroundColor: 'rgba(255, 99, 132, 0.7)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Number of Houses',
                        data: data.counts,
                        type: 'line',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: false,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                if (context.dataset.label === 'Average Price') {
                                    return `$${context.raw.toLocaleString()}`;
                                }
                                return `${context.raw} houses`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Price ($)' },
                        ticks: { callback: (val) => `$${val.toLocaleString()}` }
                    },
                    y1: {
                        position: 'right',
                        title: { display: true, text: 'Count' },
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading quality chart:', error);
    }
}

// Load feature importance
async function loadFeatureImportance() {
    try {
        const response = await fetch('http://localhost:5000/api/feature-importance');
        const data = await response.json();

        const ctx = document.getElementById('featureImportanceChart').getContext('2d');

        if (charts.featureImportanceChart) charts.featureImportanceChart.destroy();

        charts.featureImportanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.features,
                datasets: [{
                    label: 'Importance Score',
                    data: data.importance,
                    backgroundColor: 'rgba(255, 159, 64, 0.7)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: { callbacks: { label: (ctx) => ctx.raw.toFixed(4) } }
                },
                scales: {
                    x: { title: { display: true, text: 'Importance' } }
                }
            }
        });
    } catch (error) {
        console.error('Error loading feature importance:', error);
    }
}

// Load scatter plot
async function loadScatterPlot(feature) {
    try {
        const response = await fetch(`http://localhost:5000/api/scatter-data?feature=${feature}`);
        const data = await response.json();

        const ctx = document.getElementById('scatterPlot').getContext('2d');

        const scatterData = data.x.map((x, i) => ({ x: x, y: data.y[i] }));

        if (charts.scatterPlot) charts.scatterPlot.destroy();

        charts.scatterPlot = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: `Price vs ${feature}`,
                    data: scatterData,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    pointRadius: 5,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `${feature}: ${context.raw.x.toLocaleString()}, Price: $${context.raw.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: feature },
                        ticks: { callback: (val) => val.toLocaleString() }
                    },
                    y: {
                        title: { display: true, text: 'Price ($)' },
                        ticks: { callback: (val) => `$${val.toLocaleString()}` }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading scatter plot:', error);
    }
}

// Load insights
async function loadInsights() {
    try {
        // Load neighborhood insights
        const neighborhoodResponse = await fetch('http://localhost:5000/api/price-by-neighborhood');
        const neighborhoodData = await neighborhoodResponse.json();

        const topNeighborhoods = neighborhoodData.neighborhoods.slice(0, 5);
        const topPrices = neighborhoodData.prices.slice(0, 5);

        const bestNeighborhoodsHtml = topNeighborhoods.map((hood, i) =>
            `<div class="insight-item">
                <strong>${hood}</strong>: $${topPrices[i].toLocaleString()}
                <span class="badge">Top ${i+1}</span>
            </div>`
        ).join('');
        document.getElementById('best-neighborhoods').innerHTML = bestNeighborhoodsHtml;

        // Load feature importance for valuable features
        const importanceResponse = await fetch('http://localhost:5000/api/feature-importance');
        const importanceData = await importanceResponse.json();

        const topFeatures = importanceData.features.slice(0, 5);
        const topImportance = importanceData.importance.slice(0, 5);

        const valuableFeaturesHtml = topFeatures.map((feature, i) =>
            `<div class="insight-item">
                <strong>${feature.replace(/([A-Z])/g, ' $1').trim()}</strong>
                <span class="badge">Impact: ${(topImportance[i] / topImportance[0] * 100).toFixed(0)}%</span>
            </div>`
        ).join('');
        document.getElementById('valuable-features').innerHTML = valuableFeaturesHtml;

        // Price appreciation factors
        const appreciationFactors = [
            "🏗️ <strong>Higher Overall Quality</strong> - Each point increase adds ~15-20% to value",
            "📐 <strong>Larger Living Area</strong> - Every 100 sq ft adds ~$15,000",
            "🚗 <strong>Garage Capacity</strong> - Each additional garage space adds ~$10,000",
            "🛏️ <strong>More Bathrooms</strong> - Full bathroom adds ~$20,000",
            "🔥 <strong>Fireplaces</strong> - Having a fireplace adds ~$15,000"
        ];
        document.getElementById('appreciation-factors').innerHTML = appreciationFactors.map(f => `<div class="insight-item">${f}</div>`).join('');

        // Price depreciation factors
        const depreciationFactors = [
            "📅 <strong>Older Construction</strong> - Each decade older reduces value by ~5-8%",
            "🔧 <strong>Poor Condition</strong> - Below average condition reduces value by ~20%",
            "🏚️ <strong>Small Lot Size</strong> - Below median lot size reduces value by ~10%",
            "🅿️ <strong>No Garage</strong> - Reduces value by ~$15,000-20,000",
            "🌆 <strong>Undesirable Neighborhood</strong> - Can reduce value by 30-40%"
        ];
        document.getElementById('depreciation-factors').innerHTML = depreciationFactors.map(f => `<div class="insight-item">${f}</div>`).join('');

        // Investment recommendations
        const recommendations = `
            <p>🎯 <strong>Best Investment Strategy:</strong></p>
            <ul>
                <li>Focus on neighborhoods: ${topNeighborhoods.slice(0, 3).join(', ')}</li>
                <li>Prioritize houses with Overall Quality ≥ 7</li>
                <li>Look for properties with garage capacity ≥ 2 cars</li>
                <li>Consider houses built after 2000 for better appreciation</li>
                <li>Aim for living area between 1,500-2,500 sq ft for best ROI</li>
            </ul>
            <p>💰 <strong>Expected ROI:</strong> Properties in top neighborhoods with high quality ratings typically appreciate 5-8% annually.</p>
        `;
        document.getElementById('recommendations').innerHTML = recommendations;

    } catch (error) {
        console.error('Error loading insights:', error);
    }
}

// Prediction form handling
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const priceSpan = document.getElementById('price');
    const rangeSpan = document.getElementById('range');
    const errorMessageSpan = document.getElementById('errorMessage');
    const predictBtn = document.querySelector('.predict-btn');

    // Load initial statistics
    loadStatistics();

    // Scatter plot update button
    const updateScatterBtn = document.getElementById('updateScatter');
    if (updateScatterBtn) {
        updateScatterBtn.addEventListener('click', () => {
            const feature = document.getElementById('scatterFeature').value;
            loadScatterPlot(feature);
        });
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        resultDiv.style.display = 'none';
        errorDiv.style.display = 'none';

        const formData = {
            GrLivArea: parseInt(document.getElementById('GrLivArea').value),
            LotArea: parseInt(document.getElementById('LotArea').value),
            TotalBsmtSF: parseInt(document.getElementById('TotalBsmtSF').value),
            '1stFlrSF': parseInt(document.getElementById('1stFlrSF').value),
            OverallQual: parseInt(document.getElementById('OverallQual').value),
            OverallCond: parseInt(document.getElementById('OverallCond').value),
            YearBuilt: parseInt(document.getElementById('YearBuilt').value),
            YearRemodAdd: parseInt(document.getElementById('YearRemodAdd').value),
            BedroomAbvGr: parseInt(document.getElementById('BedroomAbvGr').value),
            FullBath: parseInt(document.getElementById('FullBath').value),
            HalfBath: parseInt(document.getElementById('HalfBath').value),
            Fireplaces: parseInt(document.getElementById('Fireplaces').value),
            GarageCars: parseInt(document.getElementById('GarageCars').value),
            GarageArea: parseInt(document.getElementById('GarageArea').value),
            Neighborhood: document.getElementById('Neighborhood').value,
            MSZoning: document.getElementById('MSZoning').value,
            WoodDeckSF: parseInt(document.getElementById('WoodDeckSF').value),
            OpenPorchSF: parseInt(document.getElementById('OpenPorchSF').value)
        };

        formData['2ndFlrSF'] = 0;
        formData['KitchenAbvGr'] = 1;
        formData['TotRmsAbvGrd'] = formData.BedroomAbvGr + formData.FullBath + 3;
        formData['LotFrontage'] = Math.sqrt(formData.LotArea) / 2;

        if (formData.GrLivArea < 300) {
            showError('Living area must be at least 300 sq ft');
            return;
        }

        setLoading(true);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.success) {
                priceSpan.textContent = data.formatted_price;
                rangeSpan.innerHTML = `${data.confidence_range.formatted_lower} - ${data.confidence_range.formatted_upper}`;
                resultDiv.style.display = 'block';
                priceSpan.classList.add('pulse');
                setTimeout(() => priceSpan.classList.remove('pulse'), 1000);
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
        setTimeout(() => { errorDiv.style.display = 'none'; }, 5000);
    }

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

    const yearBuilt = document.getElementById('YearBuilt');
    const yearRemod = document.getElementById('YearRemodAdd');
    yearBuilt.addEventListener('change', function() {
        if (!yearRemod.value || yearRemod.value < this.value) {
            yearRemod.value = this.value;
        }
    });
});