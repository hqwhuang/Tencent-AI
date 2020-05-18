#include <bits/stdc++.h>

using namespace std;

#define INF 2139062143
#define LINF 9187201950435737471
#define mem(a,v) memset(a,v,sizeof(a))
#define maxn 100050
#define rep(i, a, b) for (int i = (a); i <= (b); ++i)
#define red(i, a, b) for (int i = (a); i >= (b); --i)
#define pb push_back
#define eps 1e-6
#define ll long long
#define max(a,b) (a>b?a:b)
#define min(a,b) (a<b?a:b)
typedef pair<int, int> pp;

int batch_size = 250000;
//int batch_size = 25;
const int AGE_NUM = 10;
const int GENDER_NUM = 2;

struct user_profile {
    int age, gender;
    vector<int> creative_ids;
    set<int> creative_id_set;

    void add_creative_id(int creative_id) {
        if (creative_id_set.find(creative_id) == creative_id_set.end()) {
            creative_ids.pb(creative_id);
            creative_id_set.insert(creative_id);
        }
    }
};

struct creative_profile {
    vector<int> ad_ids;
    vector<int> product_ids;
    vector<int> product_categories;
    vector<int> advertiser_ids;
    vector<int> industries;
    int age_stat[11];
    int gender_stat[3];
    int pv;

    creative_profile(): pv(0) {
        mem(age_stat, 0);
        mem(gender_stat, 0);
    };

    void add_ad_ids(int aid) {
        ad_ids.pb(aid);
    }

    void add_product_id(int product_id) {
        product_ids.pb(product_id);
    }

    void add_product_category(int pc) {
        product_categories.pb(pc);
    }

    void add_advertiser_id(int aid) {
        advertiser_ids.pb(aid);
    }

    void add_industry(int iid) {
        industries.pb(iid);
    }
};

struct act_log {
    int time, user_id, creative_id, click_times;
};

map<int, user_profile> user_map;
map<int, creative_profile> creative_map;
vector<map<string, vector<int>>> record;
map<string, vector<int>> tmp;
vector<int> tmpv;
set<int> tmps;
act_log log_list[41000000];
int indexx = 1;

void filetotfrecord(int index, bool testing);

void gen_training_data(bool testing) {
    cout<<"begin to fullfill record"<<endl;
    user_map.clear();
    creative_map.clear();
    record.clear();
    ifstream user("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/user.csv");
    ifstream ad("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/adz.csv");
    ifstream click_log("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/click_log.csv");

    //read_user_map
    cout<<"begin to read user_map"<<endl;
    string line;
    getline(user, line);
    int uid,age,gender;
    while(getline(user, line)) {
        sscanf(line.c_str(), "%d,%d,%d", &uid, &age, &gender);
        user_map.insert({uid, {age, gender, tmpv, tmps}});
    }
    //read_ad_map
    cout<<"begin to read ad_map"<<endl;
    getline(ad, line);
    int creative_id, ad_id, product_id, product_category, advertiser_id, industry;
    while(getline(ad, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d,%d,%d", &creative_id, &ad_id, &product_id, &product_category, &advertiser_id, &industry);
        if (creative_map.find(creative_id) == creative_map.end()) {
            creative_map[creative_id] = creative_profile();
        }
        creative_map[creative_id].add_ad_ids(ad_id);
        creative_map[creative_id].add_product_id(product_id);
        creative_map[creative_id].add_product_category(product_category);
        creative_map[creative_id].add_advertiser_id(advertiser_id);
        creative_map[creative_id].add_industry(industry);
    }
    //calculate for stat feature
    cout<<"begin to calculate for stat feature"<<endl;
    getline(click_log, line);
    int time, user_id, click_times;
    int count = 0;
    while(getline(click_log, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d", &time, &user_id, &creative_id, &click_times);
        if (creative_map.find(creative_id) == creative_map.end()) {
            creative_map[creative_id] = creative_profile();
        }
        creative_map[creative_id].age_stat[user_map[user_id].age] += 1;
        creative_map[creative_id].gender_stat[user_map[user_id].gender] += 1;
        creative_map[creative_id].pv += 1;
//        user_map[user_id].creative_ids.pb(creative_id);
        user_map[user_id].add_creative_id(creative_id);

        log_list[count] = {time, user_id, creative_id, click_times};

        count++;
    }
    cout<<"number of log: "<<count<<endl;
    if (testing)
        return;
    cout<<"begin to generate record vector"<<endl;
    int tot = count;
    sort(log_list, log_list+tot, [](act_log &a, act_log &b){
        if(a.time == b.time)
            return a.creative_id < b.creative_id;
        return a.time < b.time;
    });
    count = 0;
    for(int h = 0; h < tot; h++) {
        time = log_list[h].time;
        user_id = log_list[h].user_id;
        creative_id = log_list[h].creative_id;
        click_times = log_list[h].click_times;
        tmp.clear();
        tmp["time"] = vector<int>(1, time);
        tmp["user_id"] = vector<int>(1, user_id);
        tmp["creative_id"] = vector<int>(1, creative_id);
        tmp["age"] = vector<int>(1, user_map[user_id].age);
        tmp["gender"] = vector<int>(1, user_map[user_id].gender);
        tmp["ad_id"] = creative_map[creative_id].ad_ids;
        tmp["product_id"] = creative_map[creative_id].product_ids;
        tmp["product_category"] = creative_map[creative_id].product_categories;
        tmp["advertiser_id"] = creative_map[creative_id].advertiser_ids;
        tmp["industry"] = creative_map[creative_id].industries;
        for (int i = 1; i <= AGE_NUM; i++) {
            tmp["age_stat"+to_string(i)] = vector<int>(1, creative_map[creative_id].age_stat[i]);
        }
        for (int i = 1; i <= GENDER_NUM; i++) {
            tmp["gender_stat"+to_string(i)] = vector<int>(1, creative_map[creative_id].gender_stat[i]);
        }
        tmp["pv"] = vector<int>(1, creative_map[creative_id].pv);
        tmp["rcid"] = user_map[user_id].creative_ids;
        record.pb(tmp);
        count++;
        if (count % batch_size==0) {
            filetotfrecord(count/batch_size, testing);
            indexx = count/batch_size;
            record.clear();
        }
    }
    filetotfrecord(indexx+1, testing);

    cout<<"finish fullfill record, with number: "<<record.size()<<endl;
}

void gen_testing_data() {
    cout<<"begin to fullfill record"<<endl;
    record.clear();
    ifstream ad("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/adz.csv");
    ifstream click_log("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/click_log.csv");
    string line;
    //read_ad_map
    cout<<"begin to read ad_map"<<endl;
    getline(ad, line);
    int creative_id, ad_id, product_id, product_category, advertiser_id, industry;
    while(getline(ad, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d,%d,%d", &creative_id, &ad_id, &product_id, &product_category, &advertiser_id, &industry);
        if (creative_map.find(creative_id) == creative_map.end()) {
            creative_map[creative_id] = creative_profile();
            creative_map[creative_id].add_ad_ids(ad_id);
            creative_map[creative_id].add_product_id(product_id);
            creative_map[creative_id].add_product_category(product_category);
            creative_map[creative_id].add_advertiser_id(advertiser_id);
            creative_map[creative_id].add_industry(industry);
        }
    }
    //calculate for stat feature
    cout<<"begin to calculate rcid"<<endl;
    getline(click_log, line);
    int time, user_id, click_times;
    int count = 0;
    while(getline(click_log, line)) {
        sscanf(line.c_str(), "%d,%d,%d,%d", &time, &user_id, &creative_id, &click_times);
        if (creative_map.find(creative_id) == creative_map.end()) {
            creative_map[creative_id] = creative_profile();
        }
        creative_map[creative_id].pv += 1;
        if (user_map.find(user_id) == user_map.end()) {
            user_map.insert({user_id, {0, 0, tmpv, tmps}});
        }
//        user_map[user_id].creative_ids.pb(creative_id);
        user_map[user_id].add_creative_id(creative_id);
        log_list[count] = {time, user_id, creative_id, click_times};

        count++;
    }
    cout<<"number of log: "<<count<<endl;
    cout<<"begin to generate record vector"<<endl;
    int tot = count;
    sort(log_list, log_list+tot, [](act_log &a, act_log &b){
        if(a.time == b.time)
            return a.creative_id < b.creative_id;
        return a.time < b.time;
    });
    count = 0;
    for (int h = 0; h < tot; h++) {
        time = log_list[h].time;
        user_id = log_list[h].user_id;
        creative_id = log_list[h].creative_id;
        click_times = log_list[h].click_times;

        tmp.clear();
        tmp["time"] = vector<int>(1, time);
        tmp["user_id"] = vector<int>(1, user_id);
        tmp["creative_id"] = vector<int>(1, creative_id);
        tmp["age"] = vector<int>(1, user_map[user_id].age);
        tmp["gender"] = vector<int>(1, user_map[user_id].gender);
        tmp["ad_id"] = creative_map[creative_id].ad_ids;
        tmp["product_id"] = creative_map[creative_id].product_ids;
        tmp["product_category"] = creative_map[creative_id].product_categories;
        tmp["advertiser_id"] = creative_map[creative_id].advertiser_ids;
        tmp["industry"] = creative_map[creative_id].industries;
        for (int i = 1; i <= AGE_NUM; i++) {
            tmp["age_stat"+to_string(i)] = vector<int>(1, creative_map[creative_id].age_stat[i]);
        }
        for (int i = 1; i <= GENDER_NUM; i++) {
            tmp["gender_stat"+to_string(i)] = vector<int>(1, creative_map[creative_id].gender_stat[i]);
        }
        tmp["pv"] = vector<int>(1, creative_map[creative_id].pv);
        tmp["rcid"] = user_map[user_id].creative_ids;
        record.pb(tmp);
        count++;
        if (count % batch_size==0) {
            filetotfrecord(count/batch_size, true);
            indexx = count/batch_size;
            record.clear();
        }
    }

    filetotfrecord(indexx+1, true);

    cout<<"finish fullfill record, with number: "<<record.size()<<endl;
}


bool replace(std::string& str, const std::string& from, const std::string& to) {
    while(true) {
        size_t start_pos = str.find(from);
        if(start_pos == std::string::npos)
            return false;
        string a = str.substr(0, start_pos);
        if (start_pos + 2 == str.length()) {
            str = a + "0";
        } else {
            str = a + "0" + str.substr(start_pos+2, str.length() - start_pos - 2);
        }
    }
}

void removeNaN(){
    ifstream ad;
    ofstream adz;
//    ad.open("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/ad.csv");
//    adz.open("/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/adz.csv");
    ad.open("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/ad.csv");
    adz.open("/Users/huangqingwei/Documents/C++ workspace/codeforces/test/adz.csv");
    string line;
    while(getline(ad, line))
    {
        replace(line, "\\N", "0");
        adz << line << '\n';
    }
    ad.close();
    adz.close();
}


void filetotfrecord(int index, bool testing){
    if(record.empty())
        return;

    cout<<"Begin to write tf example "<<index<<endl;

    string file_path = "/Users/huangqingwei/Documents/C++ workspace/codeforces/train_preliminary/training_data/train_serialize_"+to_string(index)+".csv";
    if(testing)
        file_path = "/Users/huangqingwei/Documents/C++ workspace/codeforces/test/testing_data/test_serialize_"+to_string(index)+".csv";
    ofstream output(file_path.c_str());
    //age_ratio key
    for(int i=1; i<=AGE_NUM; i++) {
        output<<"age_ratio"+to_string(i)<<',';
    }
    //gender_ratio key
    for(int i=1; i<=GENDER_NUM; i++) {
        output<<"gender_ratio"+to_string(i)<<',';
    }
    //left feature key
    int cnt = 0;
    for (auto vv : record[0]) {
        output<<vv.first;
        output<<(cnt==record[0].size()-1?'\n':',');
        cnt++;
    }
    for (auto v : record) {
        //age_ratio
        for(int i=1; i<=AGE_NUM; i++) {
            output<<1.0*v["age_stat"+to_string(i)][0]/v["pv"][0]<<',';
        }
        //gender_ratio
        for(int i=1; i<=GENDER_NUM; i++) {
            output<<1.0*v["gender_stat"+to_string(i)][0]/v["pv"][0]<<',';
        }
        //left feature
        cnt = 0;
        for (auto vv : v) {
            for(int i = 0; i < vv.second.size(); i++) {
                output<<vv.second[i];
                if(i!=vv.second.size()-1){
                    output<<':';
                }
            }
            output<<(cnt==record[0].size()-1?'\n':',');
            cnt++;
        }

    }
    output.close();
    cout<<"finish to generate tf example "<<index<<endl;
}


map<int, map<int,int>> user_age;
map<int, map<int,int>> user_gender;
int get_most(map<int, int> profile) {
    int result = 1;
    int maxx = 0;
    for (auto v : profile) {
        result = (maxx<v.second ? v.first : result);
        maxx = max(maxx, v.second);
    }
    return result;
}

void gen_result() {
    string line;
    int uid;
    int age;
    int gender;
    for(int i = 1; i < 136; i++)
        cout<<"*";
    cout<<endl;
    for(int i = 1; i < 136; i++) {
        cout<<"*"<<std::flush;
        ifstream input("/Users/huangqingwei/Documents/comp/tencentAD/predict/result_" + to_string(i) + ".txt");
        getline(input, line);
        while(getline(input, line)) {
            sscanf(line.c_str(), "%d,%d,%d", &uid, &age, &gender);
            user_age[uid][age]++;
            user_gender[uid][gender]++;
        }
    }
    ofstream output("/Users/huangqingwei/Documents/comp/tencentAD/submission.csv");
    output<<"user_id,predicted_age,predicted_gender"<<'\n';
    for (auto v : user_age) {
        output<<v.first<<','<<get_most(user_age[v.first])<<','<<get_most(user_gender[v.first])<<'\n';
    }
    output.close();
}


int main() {
    ios::sync_with_stdio(false);cout.setf(ios::fixed);cout.precision(20);
#ifdef LOCAL
//    freopen("/Users/huangqingwei/Documents/C++ workspace/codeforces/input.txt", "r", stdin);
//    freopen("/Users/huangqingwei/Documents/C++ workspace/codeforces/output.txt", "w", stdout);
#endif
    /*feature clean*/
//    removeNaN();
    /*train data*/
//    gen_training_data(false/*testing*/);
    /*test data*/
//    gen_training_data(true/*testing*/);
//    gen_testing_data();
    /*result*/
    gen_result();

#ifdef LOCAL
    cerr << "Time elapsed: " << 1.0 * clock() / CLOCKS_PER_SEC << " s.\n";
#endif
    return 0;
}














