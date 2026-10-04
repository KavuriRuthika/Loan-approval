/**
 * AI-Based Loan Approval Prediction & Financial Risk Analysis System
 * Client-Side Interactivity, Presets, and Real-Time Estimation
 */

document.addEventListener('DOMContentLoaded', () => {
    initPresets();
    initRealtimeCalculations();
    initDashboardCharts();
});

// 1. Preset Profile Configurations
const PRESETS = {
    prime: {
        Gender: 'Male',
        Married: 'Yes',
        Dependents: '1',
        Education: 'Graduate',
        Self_Employed: 'No',
        ApplicantIncome: 8500,
        CoapplicantIncome: 3000,
        LoanAmount: 180,
        Loan_Amount_Term: 360,
        Credit_History: '1.0',
        Property_Area: 'Urban'
    },
    risky: {
        Gender: 'Male',
        Married: 'No',
        Dependents: '2',
        Education: 'Not Graduate',
        Self_Employed: 'No',
        ApplicantIncome: 2200,
        CoapplicantIncome: 0,
        LoanAmount: 220,
        Loan_Amount_Term: 360,
        Credit_History: '0.0',
        Property_Area: 'Rural'
    },
    entrepreneur: {
        Gender: 'Female',
        Married: 'Yes',
        Dependents: '0',
        Education: 'Graduate',
        Self_Employed: 'Yes',
        ApplicantIncome: 7500,
        CoapplicantIncome: 4500,
        LoanAmount: 250,
        Loan_Amount_Term: 240,
        Credit_History: '1.0',
        Property_Area: 'Semiurban'
    },
    starter: {
        Gender: 'Male',
        Married: 'No',
        Dependents: '0',
        Education: 'Graduate',
        Self_Employed: 'No',
        ApplicantIncome: 4200,
        CoapplicantIncome: 1200,
        LoanAmount: 95,
        Loan_Amount_Term: 180,
        Credit_History: '1.0',
        Property_Area: 'Semiurban'
    }
};

function initPresets() {
    const presetButtons = document.querySelectorAll('.preset-chip');
    if (!presetButtons.length) return;

    presetButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const profileKey = btn.getAttribute('data-preset');
            const data = PRESETS[profileKey];
            if (!data) return;

            // Fill form inputs
            Object.keys(data).forEach(key => {
                const el = document.getElementById(key);
                if (el) {
                    el.value = data[key];
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }
            });

            // Highlight chosen preset chip
            presetButtons.forEach(b => b.style.borderColor = 'rgba(255,255,255,0.08)');
            btn.style.borderColor = '#3b82f6';
            btn.style.color = '#60a5fa';
        });
    });
}

function initRealtimeCalculations() {
    const appIncomeEl = document.getElementById('ApplicantIncome');
    const coappIncomeEl = document.getElementById('CoapplicantIncome');
    const loanAmountEl = document.getElementById('LoanAmount');
    const loanTermEl = document.getElementById('Loan_Amount_Term');

    const totalIncomeDisplay = document.getElementById('calc-total-income');
    const emiDisplay = document.getElementById('calc-emi');
    const dtiDisplay = document.getElementById('calc-dti');

    if (!appIncomeEl || !loanAmountEl || !loanTermEl) return;

    function recalculate() {
        const appInc = parseFloat(appIncomeEl.value) || 0;
        const coappInc = parseFloat(coappIncomeEl.value) || 0;
        const totalInc = appInc + coappInc;

        const loanAmt = parseFloat(loanAmountEl.value) || 0; // In thousands
        const term = parseFloat(loanTermEl.value) || 360; // In months

        const emi = (loanAmt * 1000) / (term > 0 ? term : 360);
        const dti = totalInc > 0 ? ((emi / totalInc) * 100) : 0;

        if (totalIncomeDisplay) totalIncomeDisplay.textContent = `$${totalInc.toLocaleString()}`;
        if (emiDisplay) emiDisplay.textContent = `$${Math.round(emi).toLocaleString()}/mo`;
        if (dtiDisplay) {
            dtiDisplay.textContent = `${dti.toFixed(1)}%`;
            if (dti > 45) {
                dtiDisplay.style.color = '#ef4444';
            } else if (dti > 30) {
                dtiDisplay.style.color = '#f59e0b';
            } else {
                dtiDisplay.style.color = '#10b981';
            }
        }
    }

    [appIncomeEl, coappIncomeEl, loanAmountEl, loanTermEl].forEach(el => {
        if (el) {
            el.addEventListener('input', recalculate);
            el.addEventListener('change', recalculate);
        }
    });

    recalculate();
}

function initDashboardCharts() {
    // Dynamic interactive charts if Chart.js is present
    const featCanvas = document.getElementById('chartFeatureImportance');
    if (featCanvas && window.Chart) {
        fetch('/api/feature-importance')
            .then(res => res.json())
            .then(data => {
                const labels = Object.keys(data).slice(0, 8);
                const values = labels.map(k => (data[k] * 100).toFixed(2));

                new Chart(featCanvas, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Importance Weight (%)',
                            data: values,
                            backgroundColor: 'rgba(59, 130, 246, 0.75)',
                            borderColor: '#3b82f6',
                            borderWidth: 1,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            x: {
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: { color: '#94a3b8' }
                            },
                            y: {
                                grid: { display: false },
                                ticks: { color: '#f8fafc', font: { weight: '600' } }
                            }
                        }
                    }
                });
            })
            .catch(err => console.log('Chart API error:', err));
    }
}
