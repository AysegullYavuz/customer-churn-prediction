-- ==========================================
-- Project: Enterprise Customer Churn Analysis
-- Layer: Data Engineering & Cleansing
-- Author: Aysegul Yavuz
-- Description: Cleansing Telco Churn raw data and creating optimized views for BI tools.
-- ==========================================

-- 1. Create a clean Analytics View for Power BI & Tableau
CREATE VIEW v_CleanedChurnData AS
SELECT 
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    
    -- Fix empty string issues in TotalCharges and convert to numeric (FLOAT)
    CASE 
        TRIM(TotalCharges) WHEN '' THEN 0 
        ELSE CAST(TotalCharges AS FLOAT) 
    END AS TotalCharges,
    
    -- Keep original Churn status
    Churn,
    
    -- Create a binary flag for advanced analytical calculations (1 = Churned, 0 = Active)
    CASE 
        WHEN Churn = 'Yes' THEN 1 
        ELSE 0 
    END AS ChurnBit

FROM Cleaned_Telco_Churn;
GO

-- 2. Verify the cleansed dataset
SELECT TOP 100 * FROM v_CleanedChurnData;