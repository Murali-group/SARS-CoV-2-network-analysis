
import logging
#import traceback as tb
# install suds from suds-py3
from suds import *
from suds.client import Client
#from datetime import datetime


#setup_logging()
logging.getLogger('suds.client').setLevel(logging.DEBUG)

default_url = 'https://david.ncifcrf.gov/webservice/services/DAVIDWebService?wsdl'
default_email = 'jeffl@vt.edu'


class DAVIDClient:
    def __init__(self, email=default_email, url=default_url):
        client = Client(url,timeout = 10000)
        location = 'https://david.ncifcrf.gov/webservice/services/DAVIDWebService.DAVIDWebServiceHttpSoap11Endpoint/'
        client.wsdl.services[0].setlocation(location)
        client.service.authenticate(email)
        self.client = client

    def setup_inputs(self, inputIds, idType='UNIPROT_ACCESSION', listName='gene', listType=0):
        self.client.service.addList(inputIds, idType, listName, listType)

    def setup_universe(self, universe, idType='UNIPROT_ACCESSION', listName='universe', listType=1):
        self.client.service.addList(universe, idType, listName, listType)

    def set_category(self, categoriesString='GOTERM_BP_DIRECT'):
        self.client.service.setCategories(categoriesString)

    def get_default_categories(self):
        return self.client.service.getDefaultCategoryNames()

    def build_functional_ann_chart(self, thd=0.2, ct=2):
        """
        *thd*: EASE threshold
        *ct*: count
        """
        # Get the results for the "Functional Annotation Chart"
        # use the default settings
        thd = 0.1
        ct = 2
        # TODO convert to pandas dataframe
        self.chartReport = self.client.service.getChartReport(thd,ct)
        chartRow = len(self.chartReport)
        print('Total chart records:',chartRow)
        return self.chartReport

    def write_functional_ann_chart(self, out_file):
        with open(out_file, 'w') as fOut:
            fOut.write('Category\tTerm\tCount\t%\tPvalue\tGenes\tList Total\tPop Hits\tPop Total\tFold Enrichment\tBonferroni\tBenjamini\tFDR\n')
            for simpleChartRecord in self.chartReport:
                categoryName = simpleChartRecord.categoryName
                termName = simpleChartRecord.termName
                listHits = simpleChartRecord.listHits
                percent = simpleChartRecord.percent
                ease = simpleChartRecord.ease
                Genes = simpleChartRecord.geneIds
                listTotals = simpleChartRecord.listTotals
                popHits = simpleChartRecord.popHits
                popTotals = simpleChartRecord.popTotals
                foldEnrichment = simpleChartRecord.foldEnrichment
                bonferroni = simpleChartRecord.bonferroni
                benjamini = simpleChartRecord.benjamini
                FDR = simpleChartRecord.afdr
                rowList = [categoryName,termName,str(listHits),str(percent),str(ease),Genes,str(listTotals),str(popHits),str(popTotals),str(foldEnrichment),str(bonferroni),str(benjamini),str(FDR)]
                fOut.write('\t'.join(rowList)+'\n')
            print('write file: %s. finished!' % (out_file))
