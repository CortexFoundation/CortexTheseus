#include <iostream>
#include <string.h>
#include <pthread.h>
#include "minerBot.h"
#include "gominer.h"

static vector<MinerBot*> botPool;
static unsigned int pool_idx = 0;

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

void MinerBot::stop() {
	cs.stop();
}

void MinerBot::await() {
    cs.await();
}

bool MinerBot::CuckooSolve(char *header, uint32_t header_len, uint64_t nonce, uint32_t *result, uint *result_len, uchar* target,uchar* result_hash)
{
    using std::cout;
    cs.setHeaderNonce(header, header_len, nonce);
    cs.setHashTarget(target);
    cs.solve();
    vector<cuckoo_sol> ss = cs.getSols();
    *result_len = 0;
    if (ss.size() > 0)
    {
	/*printf("sovle result : %d\n", ss.size());
    	cout << "size = " << sizeof(ss[0].data) << " PROOFSIZE = " << PROOFSIZE << "\n";
    	for (uint32_t i = 0;  i < sizeof(ss[0].data); i++) {
       		cout << ss[0].data[i] << " ";
    	}
	cout << "\n";
	*/
        memcpy(result, (uint32_t*)ss[0].data, PROOFSIZE * sizeof(uint32_t));
        memcpy(result_hash, (uchar*)ss[0].hash, 32*sizeof(uchar));
        *result_len = PROOFSIZE;
        return true;
    }
    return false;
}

bool MinerBot::CuckooSolve(const uint8_t *header, uint32_t headerLength, uint64_t nonce, vector<vector<uint32_t>>* solutions)
{
    cs.setHeaderNonce((char*)header, headerLength, nonce);
    cs.findSolutions(solutions);
    return false;
}


bool MinerBot::CuckooVerify(char *header, uint32_t header_len, uint64_t nonce,
    uint32_t *result, uchar* target, uchar* hash)
{
    cs.setHeaderNonce(header, header_len, nonce);
    cs.setHashTarget(target);
    bool ok = cs.verifySol(result, hash, target);
    return ok;
}


unsigned int getMinerBotInstance()
{
	while (true) {
        int bot_idx = pool_idx;
		if (pthread_mutex_trylock(&(botPool[bot_idx]->mutex)) == 0) {
			return bot_idx;
		}
		pool_idx = (pool_idx + 1) % botPool.size();
	}
}

void CuckooInit(uint nthread, uint32_t nInstances) {
    for (uint32_t idx = 0; idx < min(nInstances, static_cast<uint32_t>(20)) ; ++idx) {
        botPool.push_back(new MinerBot(nthread));
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

uint8_t CuckooSolve(uint8_t *header, uint32_t header_len, uint64_t nonce, result_t *result, uint32_t *result_len, uint8_t *target, uint8_t *hash) {
    uint bot_idx = getMinerBotInstance();
/*    fprintf(stdout, "nonce =%lu, target=%u, header:", nonce, target);
    for(int i = 0; i < header_len; i++)
	    fprintf(stdout, "%u ", header[i]);
*/
    uint8_t res = botPool[bot_idx]->CuckooSolve((char*)header, header_len, nonce, result, result_len, target, hash);
//    fprintf(stdout, "solve result : %d\n", res);
    CuckooRelease(bot_idx);
    return res;
}

int32_t CuckooFindSolutions(uint8_t *header, uint64_t nonce, result_t *result, uint32_t resultBuffSize, uint32_t* solLength, uint32_t *numSol) {
    uint bot_idx = getMinerBotInstance();
    vector<vector<result_t> > sols;
    uint8_t res = botPool[bot_idx]->CuckooSolve(header, 32, nonce, &sols);
    CuckooRelease(bot_idx);
    *solLength = 0;
    *numSol = sols.size();
    if (sols.size() == 0)
        return 0;
    *solLength = uint32_t(sols[0].size());
    for (size_t n = 0; n < min(sols.size(), (size_t)resultBuffSize / (*solLength)); n++)
    {
        auto& sol = sols[n];
        for (size_t i = 0; i < sol.size(); i++) {
            result[i + n * (*solLength)] = sol[i];
    //        printf(" %d", sol[i]);
        }
    //    printf("\n");
    }
    return 1;
}

unsigned char CuckooVerify(uint8_t *header, uint32_t header_len, uint64_t nonce, result_t *result, uint8_t* target, uint8_t* hash)
{
    // printf("=== uint8_t a[80] = { ");
	// for (uint32_t i = 0 ; i < header_len; i++) {
	// 	if (i != 0)
	// 		printf(",");
	// 	printf("%d", header[i]);
	// }
    // printf("};\n");
    // printf("=== uint32_t nonce = %d;", nonce);
    // printf("=== uint32_t result[42] = { ");
	// for (uint32_t i = 0 ; i < 42 ; i++) {
	// 	if (i != 0)
	// 		printf(",");
	// 	printf("%d", result[i]);
	// }
    // printf("};\n");
    // printf("=== uint8_t t[32] = { ");
	// for (uint32_t i = 0 ; i < 32 ; i++) {
	// 	if (i != 0)
	// 		printf(",");
	// 	printf("%d", target[i]);
	// }
    // printf("};\n");
    // printf("=== uint8_t h[32] = { ");
	// for (uint32_t i = 0 ; i < 32 ; i++) {
	// 	if (i != 0)
	// 		printf(",");
	// 	printf("%d", hash[i]);
	// }
    // printf("};\n");
    uint bot_idx = getMinerBotInstance();
    uint8_t res = botPool[bot_idx]->CuckooVerify((char*)header, header_len, nonce, result, target, hash);
    CuckooRelease(bot_idx);
    return res;
}
