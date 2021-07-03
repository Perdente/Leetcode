//https://leetcode.com/problems/reveal-cards-in-increasing-order/
class Solution {
public:
    vector<int> deckRevealedIncreasing(vector<int>& deck) {
        int n=deck.size();
        vector<int>ans(n);
        sort(deck.begin(),deck.end());
        deque<int>dq;
        for(int i=0;i<n;++i)dq.push_back(i);
        int flip=0,i=0;
        while(dq.size()){
            if(flip){
                dq.push_back(dq.front());
                dq.pop_front();
            }else{
                ans[dq.front()]=deck[i++];
                dq.pop_front();
            }
            flip^=1;
        }
        return ans;
    }
};
