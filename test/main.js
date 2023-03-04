function existsWithinTheLastFiveRequests(idx, websites, curEmail)
{
    count = 0
    if(idx >= 4)
    {
        range = idx-5;

        for(let i=idx; i>=range; i--)
            if(websites[i] === curEmail) ++count
        
    }
    else if(idx < 5)
    {
        for(let i=idx; i>=0; i--)
            if(websites[i] === curEmail) ++count
    }

    if(count > 2) return 429
    if(count <= 2) return 200
}

function existsWithinTheLastThirtyRequests(idx, websites, curEmail)
{
    count = 0
    if(idx >= 30)
    {
        //for example if idx = 60
        //range = 60-30
        range = idx - 30
        for(let i=idx; i>=range; i--)
            if(websites[i] === curEmail) ++count
        
    }
    if(count > 5) return 429
    if(count <= 5) return 200
}



websites = [
    "www.xyz.com",
    "www.xyz.com",
    "www.xyz.com",
    "www.abc.com",
    "www.abc.com",
    "www.xyz.com",
    "www.qqq.com",
    "www.xyz.com",
    "www.qqq.com",
    "www.xyz.com",
]

let res = [{}]

const main = () =>{

    if(websites.length < 30)
    {
        for(let i=0; i < websites.length; i++)
        {
            let output = existsWithinTheLastFiveRequests(i, websites, websites[i])
            
            if(output === 200) res.push({message:"OK",status:200})
            else if(output === 429) res.push({message:"Too many requests", status:429})
        }
    }
    
    else if(websites.length >= 30)
    {
        for(let i=0; i < websites.length; i++)
        {
            let output = existsWithinTheLastThirtyRequests(i, websites, websites[i])
            
            if(output === 200) res.push({message:"OK",status:200})
            else if(output === 429) res.push({message:"Too many requests", status:429})
        }
    }

    console.log(res)
}

res.shift()

main()