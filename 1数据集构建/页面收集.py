# -*- coding: utf-8 -*-
from lxml import etree
import re


#当前爬虫时间：xxx
#爬虫返回的数据是json串，从中取出需要的属性


def down_load(content):
    tree = etree.HTML(content)

    store_name= tree.xpath('//span[@class="J_im_icon"]/a/@title')
    print(len(store_name))
    goods_href=tree.xpath('//div[@class="p-name p-name-type-2"]/a/@href')
    print(len(goods_href))
    print(store_name)
    print(goods_href)
    return store_name,goods_href


import requests

if __name__ == '__main__':
    session = requests.session()
    # 你的账号和密码
    data = {
        "loginName": '18702534449',
        'password': '12345678A!'
    }
    # 登录界面
    url1 = 'https://passport.jd.com/new/login.aspx?ReturnUrl=https%3A%2F%2Fwww.jd.com%2F%3Fcu%3Dtrue%26utm_source%3Dbaidu-pinzhuan%26utm_medium%3Dcpc%26utm_campaign%3Dt_288551095_baidupinzhuan%26utm_term%3D0f3d30c8dba7459bb52f2eb5eba8ac7d_0_42c1177d48bd4dadb321180175afddb8'

    header = {
        "authority": "search.jd.com",
        "method": "GET",
        "path": "/Search?keyword=%E8%84%90%E6%A9%99&enc=utf-8&wq=qi%27c&pvid=9b3e4caeba234f098b28a84feb025261",
        "scheme": "https",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Cache-Control": "no-cache",
        "Cookie":"shshshfpa=8ca0c03c-e8a9-955b-9e42-a9a71ff2c221-1635127513; shshshfpx=8ca0c03c-e8a9-955b-9e42-a9a71ff2c221-1635127513; __jdu=1695263539614422899116; qrsc=3; pinId=I5cmWv6562rXgZHD7j7zXA; mt_xid=V2_52007VwMVWllbVVgbSxFbBWIGEVFUUFpaH04pXgNkAhsFCgtOD0pPTkAAM1cQTlVdBlIDThFbVzUGRgAKXwZSL0oYXwd7AxFOXlxDWxdCGVsOZwQiUG1bYlkeTxFZAFcAFVJb; areaId=21; mba_muid=1695263539614422899116; unpl=JF8EAMdnNSttX0pTBUsKHUUZQg9SW15aSERRaGdXB15YT1ZWElAcFhl7XlVdXxRLFB9sbhRXXFNJXA4eBSsSEXteXVdZDEsWC2tXVgQFDQ8VXURJQlZAFDNVCV9dSRZRZjJWBFtdT1xWSAYYRRMfDlAKDlhCR1FpMjVkXlh7VAQrAhwaFEhaVlpfAEwXBmpkBl1VUEtUAisDKxUge21WXFgPTycCX2Y1FgkESFIMHAMaXxBMVVBdWgpPFQtoZwBRXltCXA0bAhwiEXte; __jdv=70901841|baidu-pinzhuan|t_288551095_baidupinzhuan|cpc|0f3d30c8dba7459bb52f2eb5eba8ac7d_0_6061a97f88c742c2bc61cb2153b8c658|1703767199611; joyya=1703767186.1703767252.20.0n6rjli; wlfstk_smdl=5gzd0k60bg75pv4mho32a4f3cuebo5it; TrackID=1HUPq5LbqtUYj4vobDIa9k_MNWtfj5huPVSqX8mRn38Vp78SX_R_QBelT7kPNdF-WrKi59sDRNv73I78oVVTV_P37eBUJby6-3kyLPuGrdHq8Y3RfcStaxluRD_4A6eLo; thor=91DAE773FD3FCDC3A199729E061AB50870192FC36AAB63D669F90D95394E195601AC7D792D63C509F3551F69D90CFB91BE2E5423175D53368DEAEBBA2406AF787EAE38841FA3A232CD3AF6482821B5AA2582BBFBE0C8AAD684A79FA2F185C483ADB4CE6BAB52420190C31667993547B9EB697AB53979F0BF8D2D750E657E0D834414856D2D3CA5A851BB7E869C55553D; flash=2_3kVSeSvOSUfIENJ1mh-Q5qeg7tAMqLM2wSMyQxJqunsgLfxZi5hKhhYFyh6cFhHTdzQXI0cXltnvf3FdznUFTMauWf0i4-4vyCh8k6s4nzo*; pin=qwerty_11; unick=qwerty_11; ceshi3.com=000; _tp=HLIrOjSSnHQ93JtjpjhlOg%3D%3D; _pst=qwerty_11; jsavif=0; jsavif=0; rkv=1.0; ipLoc-djd=21-1827-4101-40925; __jd_ref_cls=LoginDisposition_Go; x-rp-evtoken=N-nAb5Oj6OS1u8hkvixIgG4MbzUiEHzV_3xCCn3dFpjT1fuFWVEhk_hxlMpBdlTvDe6mu-lJ37dTM3p-SvH2tBS0poPK9T-SN8MuuSgCdbVXLKpWJtUMEXKJdc2KzuO05oB8OjTTC0Yyw6kgLoQhVPJXu6f6RTKldtBAFd0LuFmcmeFku3eayHbBHvbRIW2lZ5PWRQjUR7hnOYnYu4dzPAihWDyVv528McgyTKGJ9VA%3D; xapieid=jdd03DK7VKBQYYUA3PZAIHOUMQX27RBRYWEZ37MAHD3DHI4Q6EMIZD5GFMT3CEXJPRYZWWO3T3NWWTBPE7CQ2FXRS2OFM6IAAAAMMWCRZUMIAAAAADPGGSCBLFKMQAEX; 3AB9D23F7A4B3C9B=DK7VKBQYYUA3PZAIHOUMQX27RBRYWEZ37MAHD3DHI4Q6EMIZD5GFMT3CEXJPRYZWWO3T3NWWTBPE7CQ2FXRS2OFM6I; token=aab7e5ca6a76f0743634f3e950c21bdb,3,946539; __tk=YUuEu3a5ZsuzYDlnvpk1YDrnuzTDupu0Yz2xX3kFXSJ0vSJyYzGFYG,3,946539; 3AB9D23F7A4B3CSS=jdd03DK7VKBQYYUA3PZAIHOUMQX27RBRYWEZ37MAHD3DHI4Q6EMIZD5GFMT3CEXJPRYZWWO3T3NWWTBPE7CQ2FXRS2OFM6IAAAAMMWCXIBIAAAAAADAVYLWDYFT2ZQYX; _gia_d=1; shshshsID=3a9f732f82610bd0d432fab2dd3b27dc_69_1703771273733; shshshfpb=AAsmDrrCMEqDAPOiplVueQqmnH_LCIRY1EnUTfwAAAAA; __jda=181111935.1695263539614422899116.1695263539.1703645671.1703767200.12; __jdb=181111935.77.1695263539614422899116|12.1703767200; __jdc=181111935"
    }
    header["Dnt"] = "1"
    header["Pragma"] = "no-cache"
    # header[
    #     "Referer"] = "https://search.jd.com/Search?keyword=%E8%92%9C%E5%A4%B4&qrst=1&psort=4&suggest=1.rem.0.0&wq=%E8%92%9C%E5%A4%B4&stock=1&psort=4&pvid=cd399f840e50496c9a64ced7018339d6&isList=0&page=21&s=601&click=0&log_id=1703769385022.8128"
    header["Sec-Ch-Ua"] = "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\", \"Microsoft Edge\";v=\"120\""
    header["Sec-Ch-Ua-Mobile"] = "?0"
    header["Sec-Ch-Ua-Platform"] = "\"Windows\""
    header["Sec-Fetch-Dest"] = "document"
    header["Sec-Fetch-Mode"] = "navigate"
    header["Sec-Fetch-Site"] = "same-origin"
    header["Sec-Fetch-User"] = "?1"
    header["Upgrade-Insecure-Requests"] = "1"
    header[
        "User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"

    #登录操作
    resq = session.post(url1, data=data, headers=header)

    all_store = []
    all_href = []

    # all_store =
    # all_href =
    pattern = r'\d+'
    # 变量"s"一般表示搜索结果的起始位置或页码。它用于指定搜索结果的偏移量或分页数。
    # page指定显示结果的第几页
    # 其他变量均不改变
    # 当"stock"参数的值为1时，通常表示只显示有库存的商品。这样可以过滤掉已经售罄或无货的商品，只展示可供购买的商品。。当"stock"参数的值为0时，通常表示不考虑库存状态，将所有商品都展示出来，无论是否有库存。
    # "qrst"参数的含义可能因不同的电商平台而异。在某些情况下，它可以用于指定搜索结果的排序方式。例如，当"qrst=1"时，可能表示按照默认的排序方式进行搜索展示；当"qrst=2"时，可能表示按照价格从低到高进行排序；当"qrst=3"时，可能表示按照销量从高到低进行排序，等等。
    # "click"参数通常用于记录用户的点击行为或跟踪相关统计数据。

    # 这是默认第一页的地址，没有规律。有规律的page=1设置时，也显示这页结果
    # url = "https://search.jd.com/Search?keyword=%E8%84%90%E6%A9%99&enc=utf-8&wq=qi%27c&pvid=9b3e4caeba234f098b28a84feb025261"
    #psort=3表示销量从高到低，stock默认是1改成0了
    #range是【)
    for i in range(1,120):
        print(f"当前是第{i}页")
        # j=i*30
        # print(i,j)
        url=f"https://search.jd.com/Search?keyword=%E8%84%90%E6%A9%99&qrst=1&psort=4&wq=%E8%84%90%E6%A9%99&stock=0&psort=3&pvid=9b3e4caeba234f098b28a84feb025261&isList=0&page={i}&s=0"
        resqen = session.get(url, headers=header)

        content=resqen.content.decode('utf-8')

        # with open('response.html', 'w', encoding='utf-8') as f:
        #     f.write(resqen.content.decode('utf-8'))
        store,href=down_load(content)

        for i in range(len(store)):
            if store[i] not in all_store:
                all_store.append(store[i])

                href_index = re.findall(pattern, href[i])
                all_href.extend(href_index)


        print("打印当前所有的店铺名字，当前长度是", len(all_store))
        print(all_store)
        print("打印当前所有的店铺编号，当前长度是",len(all_href))
        print(all_href)


#在第50页遇到了用户验证。把爬到的店铺编号放入all_store中，把page改成51，重新开始
#开始人工认证
#url=f"https://search.jd.com/Search?keyword=%E8%84%90%E6%A9%99&qrst=1&psort=4&wq=%E8%84%90%E6%A9%99&stock=0&psort=3&pvid=9b3e4caeba234f098b28a84feb025261&isList=0&page=51&s=0"
