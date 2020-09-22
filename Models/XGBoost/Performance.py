from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, brier_score_loss

def performance_metrics(testing_y, y_pred_binary, y_pred_rt):
    F1Macro = f1_score(testing_y, y_pred_binary, average='macro')
    F1Micro = f1_score(testing_y, y_pred_binary, average='micro')
    F1Weighted = f1_score(testing_y, y_pred_binary, average='weighted')
    PrecisionMacro = precision_score(testing_y, y_pred_binary, average='macro')
    PrecisionMicro = precision_score(testing_y, y_pred_binary, average='micro')
    PrecisionWeighted = precision_score(testing_y, y_pred_binary, average='weighted')
    RecallMacro = recall_score(testing_y, y_pred_binary, average='macro')
    RecallMicro = recall_score(testing_y, y_pred_binary, average='micro')
    RecallWeighted = recall_score(testing_y, y_pred_binary, average='weighted')
    Accuracy = accuracy_score(testing_y, y_pred_binary)
    ClassificationReport = classification_report(testing_y, y_pred_binary)
    BrierScoreProba = brier_score_loss(testing_y, y_pred_rt)
    BrierScoreBinary = brier_score_loss(testing_y, y_pred_binary)

    performance_row = {
        "F1-Macro" : F1Macro,
        "F1-Micro" : F1Micro,
        "F1-Weighted" : F1Weighted,
        "Precision-Macro" : PrecisionMacro,
        "Precision-Micro" : PrecisionMicro,
        "Precision-Weighted" : PrecisionWeighted,
        "Recall-Macro" : RecallMacro,
        "Recall-Micro" : RecallMicro,
        "Recall-Weighted" : RecallWeighted,
        "Accuracy" : Accuracy,
        "ClassificationReport" : ClassificationReport,
        "BrierScoreProba" : BrierScoreProba,
        "BrierScoreBinary" : BrierScoreBinary
    }

    return performance_row
