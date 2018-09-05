
#include <string.h>
#include <pthread.h>
#include "minerBot.h"
#include "gominer.h"
#include "cuckoo/cuckoo.h"


MinerBot::MinerBot(unsigned int nthread)
{
    if (pthread_mutex_init(&mutex, NULL) != 0) {
		printf("ERROR init with mutex");
		exit(-1);
	}
    cs.setNthreads(nthread);
    cs.initSolver();
}
MinerBot::~MinerBot()
{
    cs.release();
}

void MinerBot::testCuckoo()
{
    printf("testing cuckoo cycle...\n");
    CuckooSolver cs;
    cs.initSolver();
    char a[100];
    a[0] = '1';
    for (int i = 0; i < 64; i++)
    {
        printf("setting nonce: %d\n",i);
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

void MinerBot::stop() {
	cs.stop();
}

void MinerBot::await() {
    cs.await();
}

bool MinerBot::CuckooSolve(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uint *result_len, uchar* target,uchar* result_hash)
{
    cs.setHeaderNonce(header, header_len, nonce);
    cs.setHashTarget(target);
    cs.solve();
    vector<cuckoo_sol> ss = cs.getSols();
    *result_len = 0;
    if (ss.size() > 0)
    {
        memcpy(result, (uint32_t*)ss[0].data, PROOFSIZE * sizeof(uint32_t));
        memcpy(result_hash,(uchar*)ss[0].hash,32*sizeof(uchar));
        *result_len = PROOFSIZE;
        return true;
    }
    return false;
}


bool MinerBot::CuckooVerify(char *header, uint32_t header_len, uint32_t nonce,
    uint32_t *result, uchar* target, uchar* hash)
{
    cs.setHeaderNonce(header, header_len, nonce);
    cs.setHashTarget(target);
    bool ok = cs.verifySol(result, hash, target);
    return ok;
}



// void testGoInterface()
// {

//     printf("Welcome to Cortex Mining.\n");

//     MinerBot bot;
//     bot.testCuckoo();
// }

#define POOL_SIZE 20
static MinerBot* botPool[POOL_SIZE];
static unsigned int pool_idx = 0;
unsigned int getMinerBotInstance()
{
	while (true) {
        int bot_idx = pool_idx;
		if (pthread_mutex_trylock(&(botPool[bot_idx]->mutex)) == 0) {
			return bot_idx;
		}
		pool_idx = (pool_idx + 1) % POOL_SIZE;
	}
}

void CuckooInit(uint nthread) {
    for (auto& bot : botPool) {
        bot = new MinerBot(nthread);
    }
}

void CuckooFinalize() {
    for (auto& bot: botPool) {
        bot->stop();
    }
    for (auto& bot : botPool) {
		bot->await();
        delete bot;
    }
}

void CuckooRelease(uint bot)
{
	pthread_mutex_unlock(&(botPool[bot]->mutex));
}

unsigned char CuckooSolve(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uint *result_len, uchar* target, uchar* result_hash)
{
    uint bot_idx = getMinerBotInstance();
    uint8_t res = botPool[bot_idx]->CuckooSolve(header, header_len, nonce, result, result_len,target,result_hash);
    CuckooRelease(bot_idx);
    return res;
}
unsigned char CuckooVerify(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uchar* target, uchar* hash)
{
    uint bot_idx = getMinerBotInstance();
    uint8_t res = botPool[bot_idx]->CuckooVerify(header, header_len, nonce, result, target, hash);
    CuckooRelease(bot_idx);
    return res;
}
int  CuckooVerifyHeaderNonceAndSolutions(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result)
{
#ifndef HEADERLEN
#define HEADERLEN 80
#define HEADERLEN_TEMP_DEFINED
#endif
    char headernonce[HEADERLEN];
    memcpy(headernonce, header, header_len);
    memset(headernonce + header_len, 0, sizeof(headernonce) - header_len);
    ((u32 *)headernonce)[header_len/sizeof(u32)-1] = htole32(nonce);
    siphash_keys key;
    setheader(header, header_len, &key);
    int res = verify(result, &key);
    return res;
#ifdef HEADERLEN_TEMP_DEFINED
#undef HEADERLEN_TEMP_DEFINED
#undef HEADERLEN
#endif
}

int CuckooVerifySolutions(char *header, uint32_t header_len, uint* result)
{
    siphash_keys key;
    setheader(header, header_len, &key);
    int res = verify(result, &key);
    return res;
}

void testCuckoo(){
}
