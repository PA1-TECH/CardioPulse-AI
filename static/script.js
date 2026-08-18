document.addEventListener('DOMContentLoaded', () => {

    // 1. Theme Configuration & Toggle
    const themeBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    const currentTheme = localStorage.getItem('theme') || 'dark';
    if (currentTheme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        if (themeIcon) { themeIcon.classList.remove('fa-moon'); themeIcon.classList.add('fa-sun'); }
    }

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const isLight = document.documentElement.getAttribute('data-theme') === 'light';
            if (isLight) {
                document.documentElement.removeAttribute('data-theme');
                localStorage.setItem('theme', 'dark');
                if (themeIcon) { themeIcon.classList.remove('fa-sun'); themeIcon.classList.add('fa-moon'); }
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
                localStorage.setItem('theme', 'light');
                if (themeIcon) { themeIcon.classList.remove('fa-moon'); themeIcon.classList.add('fa-sun'); }
            }
        });
    }

    // 2. Form Submission & Prediction
    const form = document.getElementById('prediction-form');
    const placeholder = document.getElementById('results-placeholder');
    const resultsContent = document.getElementById('results-content');
    const submitBtn = document.getElementById('submit-btn');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing Clinical Profile...';

            const formData = new FormData(form);
            const patientData = {};
            formData.forEach((val, key) => {
                if (key !== 'model_choice') {
                    patientData[key] = parseFloat(val);
                }
            });

            const modelChoice = document.getElementById('model_choice').value;

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ patient: patientData, model_choice: modelChoice })
                });

                const result = await response.json();
                if (result.status === 'success') {
                    renderResults(result.data);
                    // Smooth scroll to results
                    setTimeout(() => {
                        document.getElementById('results-content')
                            ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 80);
                } else {
                    alert('Prediction Error: ' + result.message);
                }
            } catch (err) {
                console.error(err);
                alert('Failed to connect to prediction server.');
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fa-solid fa-stethoscope"></i> Predict Risk & Generate Recommendations';
            }
        });
    }

    function renderResults(data) {
        placeholder.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        const recs = data.recommendations;
        const preds = data.model_predictions;
        const sel = data.selected_prediction;

        // Risk Banner
        const riskGaugeVal = document.getElementById('risk-percentage-val');
        const riskBadge = document.getElementById('risk-category-badge');
        const urgencyText = document.getElementById('risk-urgency-text');
        const evaluatedModel = document.getElementById('evaluated-model-name');
        const riskGaugeContainer = document.querySelector('.risk-gauge-container');
        const riskBanner = document.getElementById('risk-banner');

        riskGaugeVal.textContent = `${recs.risk_percentage}%`;
        riskBadge.textContent = recs.risk_category;
        riskBadge.style.backgroundColor = recs.risk_color;
        urgencyText.textContent = recs.urgency;
        evaluatedModel.textContent = data.selected_model;

        riskGaugeContainer.style.borderColor = recs.risk_color;
        riskGaugeVal.style.color = recs.risk_color;
        riskBanner.style.borderLeftColor = recs.risk_color;

        // Individual Model Cards
        updateModelCard('dnn', preds['DNN']);
        updateModelCard('tabnet', preds['TabNet']);
        updateModelCard('mlp', preds['MLP']);

        // Recommendation Lists
        fillList('rec-medical-list', recs.medical_advice);
        fillList('rec-dietary-list', recs.dietary_advice);
        fillList('rec-bp-list', recs.bp_advice);
        fillList('rec-exercise-list', recs.exercise_advice);
        fillList('rec-monitoring-list', recs.monitoring_advice);
    }

    function updateModelCard(modelKey, predObj) {
        if (!predObj) return;
        const probElem = document.getElementById(`prob-${modelKey}`);
        const statusElem = document.getElementById(`status-${modelKey}`);

        probElem.textContent = `${predObj.risk_percentage}%`;
        statusElem.textContent = predObj.class === 1 ? 'High Risk' : 'Low Risk';

        if (predObj.class === 1) {
            statusElem.style.backgroundColor = 'rgba(239, 68, 68, 0.2)';
            statusElem.style.color = '#ef4444';
        } else {
            statusElem.style.backgroundColor = 'rgba(16, 185, 129, 0.2)';
            statusElem.style.color = '#10b981';
        }
    }

    function fillList(elementId, items) {
        const ul = document.getElementById(elementId);
        ul.innerHTML = '';
        if (!items || items.length === 0) {
            ul.innerHTML = '<li>No specific risk flags identified for this category. Maintain normal healthy routine.</li>';
            return;
        }
        items.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            ul.appendChild(li);
        });
    }

    // 3. Load Metrics for Dashboard
    async function loadMetrics() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();
            if (data.status === 'success') {
                const tbody = document.getElementById('metrics-table-body');
                tbody.innerHTML = '';
                data.metrics.forEach(m => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${m.model_name}</strong></td>
                        <td>${(m.accuracy * 100).toFixed(2)}%</td>
                        <td>${(m.precision * 100).toFixed(2)}%</td>
                        <td>${(m.recall * 100).toFixed(2)}%</td>
                        <td>${(m.specificity * 100).toFixed(2)}%</td>
                        <td>${m.f1_score.toFixed(4)}</td>
                        <td><strong>${m.roc_auc.toFixed(4)}</strong></td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        } catch (e) {
            console.error('Metrics loading error:', e);
        }
    }
});
