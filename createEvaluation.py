from openpyxl import *

#Globale Variable und Einstellungsmoeglichkeit fuer die Anzahl der Clients
#ACHTUNG: EXPONENTIALFALLE
numberOfClients = 10
path = 'Evaluation/FLStandard/'

def main(): #Erstellung von Workbook Versionen
    if input("Pls confirm the override / creation of the Evaluations Sheets (y) : ").lower() == "y":
        addVersionWorkbook('Version1')
        addVersionWorkbook('Version2')
        addVersionWorkbook('Version3')
        print("Sheets are created / replaced")
    else:
        print("Program aborted")

def addVersionWorkbook(name): #Erstellung von einem Workbook
    if (numberOfClients > 20):
        raise AssertionError

    wb = Workbook()

    addEvaluationSheet("Average loss during Training", wb)
    addEvaluationSheet('Global Test loss', wb)
    addEvaluationSheet('Global Accuracy', wb)
    for clientid1 in range(numberOfClients):
        for clientid2 in range(numberOfClients):
            addEvaluationSheet('CM ' + str(clientid2) + str(clientid1), wb)
    wb.active = wb['Global Accuracy']
    wb.save(path+name+'.xlsx')

def addEvaluationSheet(name, wb): #Erstellung von Worksheets
    ws = wb.create_sheet(name)
    wb.active = ws
    ws['A1'].value = "Average"
    #Erstellung der Überschriften
    pointer = 1
    for row in ws.iter_rows(1, 1, 2, 16):
        for cell in row:
            cell.value = "Run" + str(pointer)
            pointer += 1
    # Erstellung der Durchschnittswerte
    pointer = 2
    for row in ws.iter_rows(2, 22, 1, 1):
        for cell in row:
            cell.value = "=Average(B" + str(pointer) + ":P" + str(pointer) + ")"
            pointer += 1

if __name__ == '__main__':
    main()
