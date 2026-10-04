/**
 * AI-Based Loan Approval Prediction & Financial Risk Analysis System
 * Client-Side Interactivity, Presets, Simulation Autofill, and Real-Time Estimation
 */

document.addEventListener('DOMContentLoaded', () => {
    initSimulationTools();
    initPresets();
    initRealtimeCalculations();
    initDashboardCharts();
});

// Comprehensive Pool of Realistic Applicants for Simulation Autofill
const SIMULATION_POOL = [
    {
        name: "Prime Corporate Tech Lead",
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
    {
        name: "High Default Risk Borrower",
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
    {
        name: "Self-Employed E-Commerce Founder",
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
    {
        name: "First-Time Young Homebuyer",
        Gender: 'Male',
        Married: 'No',
        Dependents: '0',
        Education: 'Graduate',
        Self_Employed: 'No',
        ApplicantIncome: 4500,
        CoapplicantIncome: 1200,
        LoanAmount: 110,
        Loan_Amount_Term: 240,
        Credit_History: '1.0',
        Property_Area: 'Semiurban'
    },
    {
        name: "Suburban Medical Professional",
        Gender: 'Female',
        Married: 'Yes',
        Dependents: '2',
        Education: 'Graduate',
        Self_Employed: 'No',
        ApplicantIncome: 9200,
        CoapplicantIncome: 3800,
        LoanAmount: 260,
        Loan_Amount_Term: 360,
        Credit_History: '1.0',
        Property_Area: 'Semiurban'
    },
    {
        name: "Rural Agricultural Operator",
        Gender: 'Male',
        Married: 'Yes',
        Dependents: '3+',
        Education: 'Not Graduate',
        Self_Employed: 'Yes',
        ApplicantIncome: 3800,
        CoapplicantIncome: 1800,
        LoanAmount: 90,
        Loan_Amount_Term: 180,
        Credit_History: '1.0',
        Property_Area: 'Rural'
    },
    {
        name: "Over-Leveraged High Earner (Bad Credit)",
        Gender: 'Male',
        Married: 'Yes',
        Dependents: '1',
        Education: 'Graduate',
        Self_Employed: 'No',
        ApplicantIncome: 11000,
        CoapplicantIncome: 0,
        LoanAmount: 480,
        Loan_Amount_Term: 180,
        Credit_History: '0.0',
        Property_Area: 'Urban'
    },
    {
        name: "Conservative Saver (Low Debt)",
        Gender: 'Female',
        Married: 'No',
        Dependents: '0',
        Education: 'Graduate',
        Self_Employed: 'No',
        ApplicantIncome: 5200,
        CoapplicantIncome: 0,
        LoanAmount: 85,
        Loan_Amount_Term: 180,
        Credit_History: '1.0',
        Property_Area: 'Urban'
    }
];

// Preset Named Personas
const PRESETS = {
    prime: SIMULATION_POOL[0],
    risky: SIMULATION_POOL[1],
    entrepreneur: SIMULATION_POOL[2],
    starter: SIMULATION_POOL[3],
    borderline: {
        name: "Borderline Underwriting Case",
        Gender: 'Male',
        Married: 'Yes',
        Dependents: '2',
        Education: 'Graduate',
        Self_Employed: 'Yes',
        ApplicantIncome: 3900,
        CoapplicantIncome: 1100,
        LoanAmount: 160,
        Loan_Amount_Term: 360,
        Credit_History: '1.0',
        Property_Area: 'Rural'
    }
};

let lastSimIndex = -1;

/**
 * Fills the form with given applicant data and triggers animations.
 */
function autofillData(data, shouldAnimate = true) {
    if (!data) return;

    Object.keys(data).forEach(key => {
        if (key === 'name') return;
        const el = document.getElementById(key);
        if (el) {
            el.value = data[key];

            if (shouldAnimate) {
                el.classList.remove('field-autofilled');
                // Force DOM reflow for animation restart
                void el.offsetWidth;
                el.classList.add('field-autofilled');
            }

            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }
    });

    // Update Mode Status Indicator
    const modeBadge = document.getElementById('formModeBadge');
    if (modeBadge) {
        modeBadge.innerHTML = `<i class="fa-solid fa-bolt" style="color: #60a5fa;"></i> Mode: <strong>Simulation (${data.name || 'Autofilled'})</strong>`;
        modeBadge.style.borderColor = 'rgba(59, 130, 246, 0.4)';
        modeBadge.style.color = '#93c5fd';
    }
}

/**
 * Initializes Simulation, Random Autofill, and Clear Form controls.
 */
function initSimulationTools() {
    const btnRandom = document.getElementById('btnRandomSimulate');
    const btnSimAndPredict = document.getElementById('btnSimulateAndPredict');
    const btnClear = document.getElementById('btnClearForm');
    const form = document.getElementById('loanForm');

    // 1. Random Simulation Autofill Button
    if (btnRandom) {
        btnRandom.addEventListener('click', (e) => {
            e.preventDefault();
            // Pick next unique profile from pool
            let nextIdx = Math.floor(Math.random() * SIMULATION_POOL.length);
            if (nextIdx === lastSimIndex) {
                nextIdx = (nextIdx + 1) % SIMULATION_POOL.length;
            }
            lastSimIndex = nextIdx;
            const profile = SIMULATION_POOL[nextIdx];
            autofillData(profile, true);
        });
    }

    const btnFormQuick = document.getElementById('btnFormQuickAutofill');
    if (btnFormQuick) {
        btnFormQuick.addEventListener('click', (e) => {
            e.preventDefault();
            if (btnRandom) btnRandom.click();
        });
    }

    // 2. Autofill & Immediately Run Adjudication
    if (btnSimAndPredict && form) {
        btnSimAndPredict.addEventListener('click', (e) => {
            e.preventDefault();
            let nextIdx = Math.floor(Math.random() * SIMULATION_POOL.length);
            const profile = SIMULATION_POOL[nextIdx];
            autofillData(profile, false);
            // Submit form to run machine learning prediction
            form.submit();
        });
    }

    // 3. Clear Form for Manual User Entry
    if (btnClear) {
        btnClear.addEventListener('click', (e) => {
            e.preventDefault();
            const inputsToClear = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'];
            inputsToClear.forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.value = '';
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }
            });

            // Reset Selects to standard defaults
            const defaults = {
                Gender: 'Male',
                Married: 'No',
                Dependents: '0',
                Education: 'Graduate',
                Self_Employed: 'No',
                Property_Area: 'Urban',
                Loan_Amount_Term: '360',
                Credit_History: '1.0'
            };
            Object.keys(defaults).forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.value = defaults[id];
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }
            });

            // Focus primary input
            const primaryInput = document.getElementById('ApplicantIncome');
            if (primaryInput) primaryInput.focus();

            // Reset preset chip borders
            document.querySelectorAll('.preset-chip').forEach(b => {
                b.style.borderColor = 'rgba(255,255,255,0.08)';
                b.style.color = '#94a3b8';
            });

            // Update Mode Indicator
            const modeBadge = document.getElementById('formModeBadge');
            if (modeBadge) {
                modeBadge.innerHTML = `<i class="fa-solid fa-pen-to-square" style="color: #34d399;"></i> Mode: <strong>Manual Entry</strong> (Type your values)`;
                modeBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
                modeBadge.style.color = '#34d399';
            }
        });
    }
}

/**
 * Initializes quick preset profile chips.
 */
function initPresets() {
    const presetButtons = document.querySelectorAll('.preset-chip');
    if (!presetButtons.length) return;

    presetButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const profileKey = btn.getAttribute('data-preset');
            const data = PRESETS[profileKey];
            if (!data) return;

            autofillData(data, true);

            // Highlight chosen preset chip
            presetButtons.forEach(b => {
                b.style.borderColor = 'rgba(255,255,255,0.08)';
                b.style.color = '#94a3b8';
            });
            btn.style.borderColor = '#3b82f6';
            btn.style.color = '#60a5fa';
        });
    });
}

/**
 * Dynamic calculation of total income, monthly EMI, and Debt-to-Income (DTI).
 */
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

        const loanAmt = parseFloat(loanAmountEl.value) || 0; // In thousands ($)
        const term = parseFloat(loanTermEl.value) || 360; // In months

        const emi = (loanAmt > 0 && term > 0) ? ((loanAmt * 1000) / term) : 0;
        const dti = (totalInc > 0 && emi > 0) ? ((emi / totalInc) * 100) : 0;

        if (totalIncomeDisplay) {
            totalIncomeDisplay.textContent = totalInc > 0 ? `$${totalInc.toLocaleString()}` : '$0';
        }

        if (emiDisplay) {
            emiDisplay.textContent = emi > 0 ? `$${Math.round(emi).toLocaleString()}/mo` : '$0/mo';
        }

        if (dtiDisplay) {
            dtiDisplay.textContent = dti > 0 ? `${dti.toFixed(1)}%` : '0.0%';
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

/**
 * Initializes Chart.js on the analytics dashboard.
 */
function initDashboardCharts() {
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
