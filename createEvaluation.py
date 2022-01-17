from openpyxl import *

#Globale Variable und Einstellungsmoeglichkeit fuer die Anzahl der Clients
#ACHTUNG: EXPONENTIALFALLE
numberOfClients = 10
path = 'Evaluation/FLStandard/'

def main(): #Erstellung von Workbook Versionen
    for i in range (1,3):
        addVersionWorkbook('Version1')
        addVersionWorkbook('Version2')
        addVersionWorkbook('Version3')

def addVersionWorkbook(name): #Erstellung von einem Workbook
    if (numberOfClients > 20):
        raise AssertionError

    wb = Workbook()

    addEvaluationSheet("Average loss during Training", wb)
    addEvaluationSheet('Global Test loss', wb)
    addEvaluationSheet('Global Accuracy', wb)
    for clientid1 in range(numberOfClients):
        for clientid2 in range(numberOfClients):
            addEvaluationSheet('CM ' + str(clientid1) + str(clientid2), wb)

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

if __name__ == '__main__':
    main()
