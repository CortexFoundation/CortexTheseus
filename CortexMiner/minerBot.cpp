#include "minerBot.h"
#include "zcash/equihash.hpp"
// #include "cuckoo/cuckooSolver.h"
#include "gominer.h"
char modestr[][20] = {
    "equihash",
    "ethash",
    "cuckoocycle",
    "cryptonight"};

void minerBot::loadParam()
{
    XMLDocument cfg;
    cfg.LoadFile("config.xml");
    Tinyxml_Reader reader;
    reader.Use(cfg.FirstChildElement("param"));
    mode = reader.GetInt("mode");

    printf("mining mode: %s\n", modestr[mode]);
}

void minerBot::start()
{
    printf("start minging...\n");
}

void minerBot::testEquihash()
{
    printf("testing equihash...\n");
    equiSolver es;
    es.getHeader(108, NULL);
    es.initOpenCL();
    es.runOpenCL();
    //es.release();
}

void minerBot::testCuckoo()
{
    printf("testing cuckoo cycle...\n");
    cuckooSolver cs;
    cs.initSolver();
    char a[100];
    a[0] = '1';
    for (int i = 0; i < 64; i++)
    {

        cs.setHeaderNonce(a, 80, i);
        cs.solve();
        printf("%d sols\n", cs.getNumSols());
        vector<cuckoo_sol> ss = cs.getSols();
        for (int i = 0; i < ss.size(); i++)
        {
            for (int j = 0; j < PROOFSIZE; j++)
            {
                printf(" %jx", (uintmax_t)ss[i].data[j]);
            }
            printf("\n");
        }
    }
    cs.release();
}

void minerBot::CuckooInit()
{
    if (cs == NULL)
    {
        cs = new cuckooSolver();
        cs->initSolver();
        printf("init cuckoo complete");
    }
}
void minerBot::CuckooSolve(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uint *result_len)
{
    printf("%s %d %d\n",header,header_len,nonce);
    cs->setHeaderNonce(header, header_len, nonce);
    cs->solve();
    vector<cuckoo_sol> ss = cs->getSols();
    *result_len = 0;
    if (ss.size() > 0)
    {
        memcpy(result, (uint32_t*)ss[0].data, PROOFSIZE * sizeof(uint32_t));

        *result_len = PROOFSIZE;
    }
}
bool minerBot::CuckooVerify(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result)
{
    printf("%s %d %d\n",header,header_len,nonce);
    cs->setHeaderNonce(header, header_len, nonce);
    bool ok = cs->verifySol(result);
    return ok;
}
void minerBot::CuckooRelease()
{
    cs->release();
}

void testGoInterface()
{

    printf("Welcome to Cortex Mining.\n");

    minerBot bot;
    //bot.loadParam();
    //bot.start();

    //bot.testEquihash();
    bot.testCuckoo();
}

minerBot *bot;
void CuckooInit()
{
    if (bot == NULL)
    {
        bot = new minerBot();
    }
    bot->CuckooInit();
}
void CuckooSolve(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uint *result_len)
{
    // return NULL;
    // bot->CuckooSolve(header, header_len, nonce, result, result_len);
	bot->CuckooSolve(NULL, 80, 63, result, result_len);
}
unsigned char CuckooVerify(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result)
{
    // return NULL;
    // return bot->CuckooVerify(header, header_len, nonce, result);
	return bot->CuckooVerify(NULL, 80, 63, result);
}
void CuckooRelease()
{
    bot->CuckooRelease();
}
